from services.summary_storage import save_summary, load_summaries


print("Testing summary storage...")


# Save some test summaries
save_summary(1, "This is the summary of chunk 1.")
save_summary(2, "This is the summary of chunk 2.")
save_summary(3, "This is the summary of chunk 3.")


print("\nSaved summaries:")

summaries = load_summaries()

for chunk_number, summary in summaries.items():
    print(f"Chunk {chunk_number}: {summary}")


print("\nStorage test completed.")