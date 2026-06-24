from pathlib import Path
import subprocess

#this subprocess script allows the Microportraits script to be run over a whole folder.

#it is needed to set the folder information in the script itself, this is not a variable that can be defined in the command line.
input_folder = Path("data/articles_syntax/naf")
output_folder = Path("data/articles_syntax/microportraits_syntax_out")

output_folder.mkdir(exist_ok=True)

for input_file in input_folder.glob("1*"):

    output_file = output_folder / f"{input_file.stem}_out.csv"

    # by using subprocess, it is possible to run a script from within another script.
    with open(output_file, "w") as outfile:
        subprocess.run(
            ["python", "microportraits_old.py", str(input_file)],
            stdout=outfile,
            text=True,
            check=True
        )