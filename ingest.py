from rag import build_vector_database

def main():
    print("=" * 60)
    print("BUILDING RAG KNOWLEDGE BASE")
    print("=" * 60)
    try:
        count = build_vector_database()
        print(f"\nSuccessfully indexed {count} document chunks.")
        print("\nKnowledge base is ready!")
    except Exception as error:
        print(f"\nERROR:\n{error}")

if __name__ == "__main__":
    main()
