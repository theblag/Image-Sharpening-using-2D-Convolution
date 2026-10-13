#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

inline uint8_t clamp_pixel(int val) {
    return static_cast<uint8_t>(std::max(0, std::min(255, val)));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 5) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <height> <width>\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int H = std::stoi(argv[3]);
    int W = std::stoi(argv[4]);

    // 1. Calculate row distribution across ranks (handling non-divisible heights)
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int base_rows = H / size;
    int rem_rows = H % size;

    for (int i = 0; i < size; ++i) {
        int r_count = base_rows + (i < rem_rows ? 1 : 0);
        sendcounts[i] = r_count * W;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    int local_rows = sendcounts[rank] / W;

    // 2. Rank 0 reads the full input file
    std::vector<uint8_t> full_input;
    if (rank == 0) {
        full_input.resize(static_cast<size_t>(H) * W);
        std::ifstream fin(input_path, std::ios::binary);
        if (!fin) {
            std::cerr << "Rank 0: Failed to open input file " << input_path << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fin.read(reinterpret_cast<char*>(full_input.data()), full_input.size());
        fin.close();
    }

    // Allocate local memory: local_rows + 2 ghost rows (top and bottom)
    std::vector<uint8_t> local_buf((local_rows + 2) * W, 0);
    // Buffer for sharpened output
    std::vector<uint8_t> local_out(local_rows * W, 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // 3. Scatter input row blocks to local_buf starting at row index 1 (leaving index 0 for top ghost row)
    MPI_Scatterv(
        rank == 0 ? full_input.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_BYTE,
        &local_buf[1 * W],
        local_rows * W,
        MPI_BYTE,
        0,
        MPI_COMM_WORLD
    );

    // 4. Exchange boundary rows (Halo / Ghost Cells)
    int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // Send local first row up, receive bottom ghost row from below
    MPI_Sendrecv(
        &local_buf[1 * W], W, MPI_BYTE, prev_rank, 0,
        &local_buf[(local_rows + 1) * W], W, MPI_BYTE, next_rank, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Send local last row down, receive top ghost row from above
    MPI_Sendrecv(
        &local_buf[local_rows * W], W, MPI_BYTE, next_rank, 1,
        &local_buf[0], W, MPI_BYTE, prev_rank, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Topmost rank has no upper neighbor: zero-pad top halo row
    if (rank == 0) {
        std::fill(local_buf.begin(), local_buf.begin() + W, 0);
    }
    // Bottommost rank has no lower neighbor: zero-pad bottom halo row
    if (rank == size - 1) {
        std::fill(local_buf.begin() + (local_rows + 1) * W, local_buf.end(), 0);
    }

    // 5. Local 2D Convolution using 3x3 Laplacian Sharpening Kernel
    for (int r = 1; r <= local_rows; ++r) {
        for (int c = 0; c < W; ++c) {
            int center = local_buf[r * W + c];
            int up     = local_buf[(r - 1) * W + c];
            int down   = local_buf[(r + 1) * W + c];
            int left   = (c > 0)     ? local_buf[r * W + (c - 1)] : 0; // zero-padding border
            int right  = (c < W - 1) ? local_buf[r * W + (c + 1)] : 0; // zero-padding border

            int val = 5 * center - (up + down + left + right);
            local_out[(r - 1) * W + c] = clamp_pixel(val);
        }
    }

    // 6. Gather all computed slices back to Rank 0
    std::vector<uint8_t> full_output;
    if (rank == 0) {
        full_output.resize(static_cast<size_t>(H) * W);
    }

    MPI_Gatherv(
        local_out.data(),
        local_rows * W,
        MPI_BYTE,
        rank == 0 ? full_output.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    // 7. Rank 0 writes output and logs timing
    if (rank == 0) {
        std::ofstream fout(output_path, std::ios::binary);
        if (!fout) {
            std::cerr << "Rank 0: Failed to open output file " << output_path << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fout.write(reinterpret_cast<char*>(full_output.data()), full_output.size());
        fout.close();

        std::cout << "SUCCESS," << (t_end - t_start) << std::endl;
    }

    MPI_Finalize();
    return 0;
}
