from services.summary_storage import save_summary, load_summaries


def process_chunks(total_chunks, fail_at=None):

    saved_summaries = load_summaries()

    print("\nStarting processing...")

    for i in range(1, total_chunks + 1):

        # Skip already completed chunks
        if str(i) in saved_summaries:
            print(f"Chunk {i}: SKIPPED (already saved)")
            continue

        print(f"Chunk {i}: PROCESSING...")

        # Simulate failure
        if i == fail_at:
            print(f"Chunk {i}: FAILED!")
            break

        summary = f"Summary generated for chunk {i}"

        save_summary(i, summary)

        print(f"Chunk {i}: SAVED")


print("===== FIRST RUN =====")

process_chunks(total_chunks=5, fail_at=4)


print("\n===== SECOND RUN =====")

process_chunks(total_chunks=5)