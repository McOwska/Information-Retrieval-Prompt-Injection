from llm import call_llm


def main():
    response = call_llm(
        messages=[
            {
                "role": "user",
                "content": "Return only the words: hello world"
            }
        ],
        temperature=0.0,
        max_tokens=64,
    )

    print("RESPONSE:")
    print(response)


if __name__ == "__main__":
    main()