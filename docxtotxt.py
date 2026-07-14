import os
from docx import Document

INPUT_FOLDER  = "."   # <-- change this
OUTPUT_FOLDER = "output"  # <-- change this

def docx_to_txt(input_folder, output_folder):
    """Recursively convert all .docx files to .txt, mirroring folder structure."""

    found = 0
    converted = 0
    failed = 0

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if not filename.endswith('.docx'):
                continue

            found += 1
            docx_path = os.path.join(root, filename)

            relative_dir = os.path.relpath(root, input_folder)
            output_dir   = os.path.join(output_folder, relative_dir)
            os.makedirs(output_dir, exist_ok=True)

            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path     = os.path.join(output_dir, txt_filename)

            try:
                doc  = Document(docx_path)
                text = '\n'.join([para.text for para in doc.paragraphs])

                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)

                print(f"✅ {os.path.join(relative_dir, filename)}")
                print(f"    → {txt_path}\n")
                converted += 1

            except Exception as e:
                print(f"❌ Failed: {docx_path}\n    Error: {e}\n")
                failed += 1

    print("─" * 50)
    print(f"Found: {found} | Converted: {converted} | Failed: {failed}")
    print("─" * 50)

if __name__ == "__main__":
    if not os.path.isdir(INPUT_FOLDER):
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
    else:
        print(f"📂 Scanning: {INPUT_FOLDER}")
        print(f"💾 Output:   {OUTPUT_FOLDER}\n")
        docx_to_txt(INPUT_FOLDER, OUTPUT_FOLDER)