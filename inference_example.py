import argparse

from llama import (
    load_llama_from_checkpoint,
    generate_token_sequence,
)

def main(args: argparse.Namespace) -> None:
    llama, tokenizer = load_llama_from_checkpoint(
        args.model_directory,
        args.tokenizer
    )

    context: str = args.context

    prompt = tokenizer.encode(context, True, False)

    print(f'Prompt: {context}')
    print(f'Generated: ', end='')

    for token in generate_token_sequence(llama,
                                         prompt,
                                         top_p = args.top_p,
                                         max_generation_length = args.max_generation_length):
      piece = tokenizer.id_to_piece(token)
      print(piece, end='', flush=True)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using a Llama model.')

    parser.add_argument('model_directory', type=str,
                        help='Path to the directory containing the Llama model.')
    parser.add_argument('tokenizer', type=str,
                        help='Path to the tokenizer model file.')
    parser.add_argument('--context', type=str, default='Hello, world!',
                        help='Initial context to seed the Llama model.')
    parser.add_argument('--max_generation_length', type=int, default=None,
                        help='Maximum length of the generated sequence.')
    parser.add_argument('--top_p', type=float, default=0.80,
                        help='The cumulative distribution function (CDF) to use for sampling.')

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    main(args)
