import re
from pathlib import Path
from subprocess import run, PIPE, STDOUT

def extract_parameters(path: Path) -> dict | None:
  NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
  
  pattern = re.compile(
    rf"m_D=(?P<D>{NUMBER})"
    rf"_Ms=(?P<Ms>{NUMBER})"
    rf"_T=(?P<T>{NUMBER})"
    rf"_dmi=(?P<dmi>{NUMBER})"
    rf"_Ku=(?P<Ku>{NUMBER})"
    rf"(?:_RKKY=(?P<RKKY>{NUMBER}))?"
    rf"\.ovf$"
  )
  
  match = pattern.fullmatch(path.name)

  if match is None:
    return None

  params = {
    key: float(value)
    for key, value in match.groupdict().items()
    if value is not None
  }

  params.setdefault("RKKY", -1.5)
  params["path"] = str(path)

  return params

def main():

  scriptfile = Path("saf_energies.txt")
  ovf_files = Path("../ovf_files/saf_results_relax")
  
  for path in ovf_files.glob("dmi=*/*.ovf"):
    params = extract_parameters(path)
    
    relaxed_path = Path(params['path'])
    table = Path("../results/data/saf_results_relax") / Path(re.sub(r"^m_", "table_", path.name)).with_suffix(".txt")
    
    with open('mumax_templates/saf_relax_get_energy_template.txt', 'r') as file:
      saf_script = file.read()

    with open(scriptfile, 'w') as f:
      f.write(saf_script.format(
        Msat=params['Ms'],
        D=params['D'],
        T=0,
        DMI=params['dmi'],
        Ku=params['Ku'],
        rkky=params['RKKY'],
        relaxed_ovf=relaxed_path.as_posix()
      ))
    
    run(["mumax3","-f",scriptfile], stdout=PIPE, stderr=STDOUT)

    with open('saf_energies.out/table.txt', 'r') as input:
      output = open(table, 'w')
      output.write(input.read())

    print(f"Updated energies for {relaxed_path.as_posix()}")

if __name__ == '__main__':
  main()