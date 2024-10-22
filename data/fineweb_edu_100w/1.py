import sys

def repeat_jsonl(input_file, output_file, repeat_count):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        content = infile.read()
        for _ in range(repeat_count):
            outfile.write(content)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python repeat_jsonl.py <input_file> <output_file> <repeat_count>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    repeat_count = int(sys.argv[3])

    repeat_jsonl(input_file, output_file, repeat_count)
    print(f"File content repeated {repeat_count} times. Output saved to {output_file}")