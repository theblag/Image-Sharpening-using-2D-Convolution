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

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 5) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <height> <width>\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int H = std::stoi(argv[3]);
    int W = std::stoi(argv[4]);

    // 1. Create a 1D Cartesian Topology
    int dims[1] = {world_size};
    int periods[1] = {0}; // Non-periodic (boundaries do not wrap)
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart_comm);

    int rank;
    MPI_Comm_rank(cart_comm, &rank);

    int prev_rank, next_rank;
    MPI_Cart_shift(cart_comm, 0, 1, &prev_rank, &next_rank);

    // 2. Row distribution calculations (handling uneven heights)
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);
    int base_rows = H / world_size;
    int rem_rows = H % world_size;

    for (int i = 0; i < world_size; ++i) {
        int r_count = base_rows + (i < rem_rows ? 1 : 0);
        sendcounts[i] = r_count * W;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    int local_rows = sendcounts[rank] / W;

    // 3. Rank 0 reads full image from disk
    std::vector<uint8_t> full_input;
    if (rank == 0) {
        full_input.resize(static_cast<size_t>(H) * W);
        std::ifstream fin(input_path, std::ios::binary);
        if (!fin) {
            std::cerr << "Rank 0: Failed to open input " << input_path << "\n";
            MPI_Abort(cart_comm, 1);
        }
        fin.read(reinterpret_cast<char*>(full_input.data()), full_input.size());
        fin.close();
    }

    // Allocate local buffer: local_rows + 2 ghost rows (row 0: top halo, row local_rows+1: bottom halo)
    std::vector<uint8_t> local_buf((local_rows + 2) * W, 0);
    std::vector<uint8_t> local_out(local_rows * W, 0);

    MPI_Barrier(cart_comm);
    double t_start = MPI_Wtime();

    // 4. Scatter data to ranks starting at offset row 1
    MPI_Scatterv(
        rank == 0 ? full_input.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_BYTE,
        &local_buf[1 * W],
        local_rows * W,
        MPI_BYTE,
        0,
        cart_comm
    );

    // 5. Non-blocking Halo Exchange for compute-communication overlap
    MPI_Request reqs[4];
    int req_count = 0;

    // Post non-blocking receives for ghost rows
    MPI_Irecv(&local_buf[0], W, MPI_BYTE, prev_rank, 100, cart_comm, &reqs[req_count++]);
    MPI_Irecv(&local_buf[(local_rows + 1) * W], W, MPI_BYTE, next_rank, 101, cart_comm, &reqs[req_count++]);

    // Post non-blocking sends for boundary rows
    MPI_Isend(&local_buf[1 * W], W, MPI_BYTE, prev_rank, 101, cart_comm, &reqs[req_count++]);
    MPI_Isend(&local_buf[local_rows * W], W, MPI_BYTE, next_rank, 100, cart_comm, &reqs[req_count++]);

    // 6. OVERLAP: Immediately compute inner rows (rows 2 to local_rows - 1)
    for (int r = 2; r < local_rows; ++r) {
        for (int c = 0; c < W; ++c) {
            int center = local_buf[r * W + c];
            int up     = local_buf[(r - 1) * W + c];
            int down   = local_buf[(r + 1) * W + c];
            int left   = (c > 0)     ? local_buf[r * W + (c - 1)] : 0;
            int right  = (c < W - 1) ? local_buf[r * W + (c + 1)] : 0;

            int val = 5 * center - (up + down + left + right);
            local_out[(r - 1) * W + c] = clamp_pixel(val);
        }
    }

    // 7. Synchronize boundary communication
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // Zero-padding on outer boundaries
    if (rank == 0) {
        std::fill(local_buf.begin(), local_buf.begin() + W, 0);
    }
    if (rank == world_size - 1) {
        std::fill(local_buf.begin() + (local_rows + 1) * W, local_buf.end(), 0);
    }

    // 8. Compute boundary row 1
    if (local_rows >= 1) {
        int r = 1;
        for (int c = 0; c < W; ++c) {
            int center = local_buf[r * W + c];
            int up     = local_buf[(r - 1) * W + c];
            int down   = (local_rows > 1) ? local_buf[(r + 1) * W + c] : 0;
            int left   = (c > 0)     ? local_buf[r * W + (c - 1)] : 0;
            int right  = (c < W - 1) ? local_buf[r * W + (c + 1)] : 0;

            int val = 5 * center - (up + down + left + right);
            local_out[(r - 1) * W + c] = clamp_pixel(val);
        }
    }

    // 9. Compute boundary row local_rows (if local_rows > 1)
    if (local_rows > 1) {
        int r = local_rows;
        for (int c = 0; c < W; ++c) {
            int center = local_buf[r * W + c];
            int up     = local_buf[(r - 1) * W + c];
            int down   = local_buf[(r + 1) * W + c];
            int left   = (c > 0)     ? local_buf[r * W + (c - 1)] : 0;
            int right  = (c < W - 1) ? local_buf[r * W + (c + 1)] : 0;

            int val = 5 * center - (up + down + left + right);
            local_out[(r - 1) * W + c] = clamp_pixel(val);
        }
    }

    // 10. Gather all local slices back to Rank 0
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
        cart_comm
    );

    MPI_Barrier(cart_comm);
    double t_end = MPI_Wtime();

    if (rank == 0) {
        std::ofstream fout(output_path, std::ios::binary);
        if (!fout) {
            std::cerr << "Rank 0: Failed to open output file " << output_path << "\n";
            MPI_Abort(cart_comm, 1);
        }
        fout.write(reinterpret_cast<char*>(full_output.data()), full_output.size());
        fout.close();

        std::cout << "SUCCESS," << (t_end - t_start) << std::endl;
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
