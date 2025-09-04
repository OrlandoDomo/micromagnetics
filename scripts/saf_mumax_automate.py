import discretisedfield as df
import imageio.v3 as iio
import matplotlib.pyplot as plt

import os
import re

from io import BytesIO
from subprocess import run, PIPE, STDOUT

SCRIPT_SAF = """
D := {D}
setgridsize(D/3,D/3,3)
setcellsize(3e-9,3e-9,1e-9) 
//setpbc(2,2,0)

// Parameters

Msat = {Msat}e3 // A/m 
Aex = 1e-11 // J/m 
Ku1 = 0.1e6 // J/m^3
AnisU = vector(0,0,1)  // easy axis along z-direction
alpha = 0.3 
Dind = 1e-3 // J/m^2

// Custom Fields implementation for exchange between the 2 FM layers 

cellsize := 1e-9
AFMAex := -1.5e-13 // J/m
Ms := {Msat}e3

spacerthickness := 1

prefactorZ := Const( (2 * AFMAex) / ( (spacerthickness+1)*cellsize*(spacerthickness+1)*cellsize*Ms))

up := Mul(prefactorZ, Mul(Add(Mul(Const(-1),m),Shifted(m,0,0,2)),Shifted(Const(1),0,0,2)))
down := Mul(prefactorZ, Mul(Add(Mul(Const(-1),m),Shifted(m,0,0,-2)),Shifted(Const(1),0,0,-2)))

Bc := Add(up,down)

AddFieldTerm(Bc)
addEdensTerm(Mul(Const(-0.5),Dot(Bc,M_full)))

// define 2 layers
defregion(1, layer(2)) // top layer
defregion(2, layer(0)) // bottom layer

// set geometry
Setgeom(layer(0).add(layer(2)).Intersect(cylinder({D}e-9,3e-9)))


// Define initial magnetization

inner_top := cylinder(40e-9,1e-9).transl(0,0,0.5e-9)
inner_bottom := cylinder(40e-9,1e-9).transl(0,0,-0.5e-9)

m.setinshape(inner_top, uniform(0,0,1).transl(0,0,0.5e-9))
m.setinshape(inner_bottom, uniform(0,0,-1).transl(0,0,-0.5e-9))

outer_top := cylinder({D}e-9,1e-9).transl(0,0,0.5e-9).sub(inner_top)
outer_bottom := cylinder({D}e-9,1e-9).transl(0,0,-0.5e-9).sub(inner_bottom)

m.setinshape(outer_top, uniform(0,0,-1).transl(0,0,0.5e-9))
m.setinshape(outer_bottom, uniform(0,0,1).transl(0,0,-0.5e-9))

//saveas(m, "inital_mag")

relax()

//saveas(m, "relaxed_state")

Bmax  := 4.0
Bstep :=  1.0e-1
MinimizerStop = 1e-6
TableAdd(B_ext)

for B:=0.0; B<=Bmax; B+=Bstep{{
    B_ext = vector(0, 0, B)
    minimize()
    save(m)
    tablesave()
}}

for B:=Bmax; B>=-Bmax; B-=Bstep{{
    B_ext = vector(0, 0, B)
    minimize()
    save(m)
    tablesave()
}}

for B:=-Bmax; B<=Bmax; B+=Bstep{{
    B_ext = vector(0, 0, B)
    minimize()
    save(m)
    tablesave()
}}

"""

HVALUES = list(range(0,40,1)) + list(range(40,-40,-1)) + list(range(-40,40,1))

def find_ovf_files(driver_path):
  dir_list = os.listdir(driver_path)
  r = re.compile(".*ovf")
  ovf_file_path = list(filter(r.match, dir_list))

  return ovf_file_path

def create_gif_saf(D, Ms):
  
  ovf_files = find_ovf_files('saf_mumax.out')
  
  images = list()
  
  for i, ovf_file in enumerate(ovf_files):
    read_field = df.Field.from_file(f'saf_mumax.out/{ovf_file}')
    fig, axs = plt.subplots(
        figsize=(12, 6),
        nrows=1,
        ncols=2
    )
    read_field.sel(z=0.5e-9).z.mpl.scalar(ax=axs[0],cmap='coolwarm')
    read_field.sel(z=0.5e-9).resample((25, 25)).mpl.vector(
        ax=axs[0], use_color=False, color="black"
    )
    read_field.sel(z=2.5e-9).z.mpl.scalar(ax=axs[1],cmap='coolwarm')
    read_field.sel(z=2.5e-9).resample((25, 25)).mpl.vector(
        ax=axs[1], use_color=False, color="black"
    )
    
    axs[0].set_title(r"Bottom Layer: $z = 0.5 \times 10^{-9}$ m")
    axs[1].set_title(r"Top Layer: $z = 2.5 \times 10^{-9}$ m")
    
    fig.suptitle(
        rf"H={HVALUES[i]/10:.1f} T",
        fontsize='xx-large'
    )
    fig.tight_layout()
    buffer = BytesIO()

    fig.savefig(buffer, format='png')
    buffer.seek(0)
    
    images.append(iio.imread(buffer))
    plt.close(fig)

  iio.imwrite(f'../images/mumax_run/saf_skyrmion_hysteresis-{D}nm-{Ms}kA_m.gif', images, fps=2)

def run_main(D, Ms):
  scriptfile = 'saf_mumax.txt'

  with open(scriptfile, 'w') as f:
    f.write(SCRIPT_SAF.format(
        Msat=Ms,
        D=D
    ))

  run(["mumax3","-f",scriptfile], stdout=PIPE, stderr=STDOUT)
  
  create_gif_saf(D, Ms)

if __name__ == '__main__':
  Ds = range(150, 825, 75)
  Mss = range(260, 460, 20)

  for D in Ds:
    for Ms in Mss:
      try:
        run_main(D, Ms)
        print(f'Finished job for D={D} nm and Msat={Ms} kA/m')
      except:
        print(f'Could not finish job for D={D} nm and Msat={Ms} kA/m')