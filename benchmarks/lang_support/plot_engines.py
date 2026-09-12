"""Compatibility entry point for shared language benchmark reporting."""
import argparse
from pathlib import Path

from reporting import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("bn", "pt-BR", "all"), default="bn")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    languages = ("bn", "pt-BR") if args.language == "all" else (args.language,)
    for language in languages:
        results_dir = (Path(args.results_dir)
                       if args.results_dir and len(languages) == 1
                       else base / "results" / language)
        if args.out_dir:
            out_dir = Path(args.out_dir) / language if len(languages) > 1 else Path(args.out_dir)
        else:
            out_dir = base / "results" / language
        report(results_dir, out_dir, f"{language} ASR engine comparison",
               show=not args.no_show)


if __name__ == "__main__":
    main()
