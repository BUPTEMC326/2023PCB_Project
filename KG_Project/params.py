# 时钟晶振
f_clk=1e10
I_clk=1
A_clk=1
sin_theta_clk=0.7
r_clk=0.1





# 预留ESD保护器件（二极管、MOS\BJT）
diode=0
MOS=0
BJT=0
ESDProDev=bool(diode|MOS|BJT)


# 滤波器件（PCB锡球，TVS（瞬态抑制二极管），电容，电感，CMC（共模电感），FB（磁珠），GDT（气体放电管），TSS（半导体放电管），MOV（压敏电阻））。
solderBall=0
TSVDev=0
cap=0
induct=0
CMC=0
FB=0
GDT=0
TSS=0
MOV=0
filterDev=bool(solderBall | TSVDev | cap | induct | CMC| FB | GDT| TSS | MOV)

# IO口NC pin（not connect）空脚要预留TVS，或接地.
NCpin={'tvs':0}



