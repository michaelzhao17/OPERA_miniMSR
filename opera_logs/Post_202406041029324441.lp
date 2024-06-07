Opera-3d Post-Processor                 started on 04/Jun/2024 at 10:29:32
Opera 2020 SP3 [Build 2]
(c) Copyright Dassault Systemes Ltd 1984-2019. All Rights Reserved.
Node: TUCAN-MAGPC. Processor: Intel64. System: Windows 10
Number of threads = 8. Available processors (CPU cores) = 8.
Command language initialization.
Unit of Length           : cm                | Unit of Magn Flux Density: gauss
Unit of Magnetic Field   : oersted           | Unit of Magn Scalar Pot  : oersted cm
Unit of Magn Vector Pot  : gauss cm          | Unit of Conductivity     : S/cm
Unit of Current Density  : A/cm^2            | Unit of Power            : W
Unit of Force            : N                 | Unit of Energy           : J
Unit of Electric Field   : V/cm              | Unit of Elec Flux Density: C/cm^2
Unit of Mass             : g                 | Unit of Pressure         : Pa
Unit of Charge Density   : C/cm^3            | Unit of Electric Pot     : volt
Information: opera.comi was not found in the local or home directory
**** COMMAND INPUT *** ACTIVATE FILE='C:\Users\mzhao\OPERA_miniMSR\miniMSR_layer.op3'
Opening file for checking: C:\Users\mzhao\OPERA_miniMSR\miniMSR_layer.op3
**** COMMAND INPUT *** LOAD
Attaching file as resident: C:\Users\mzhao\OPERA_miniMSR\miniMSR_layer.op3
Opening database ...zhao\OPERA_miniMSR\miniMSR_layer.op3, simulation number 1 on 04/Jun/2024 at 10:29:33
Unit of Length           : m                 | Unit of Magn Flux Density: T
Unit of Magnetic Field   : A/m               | Unit of Magn Scalar Pot  : A
Unit of Magn Vector Pot  : Wb/m              | Unit of Conductivity     : S/m
Unit of Current Density  : A/m^2             | Unit of Power            : W
Unit of Force            : N                 | Unit of Energy           : J
Unit of Electric Field   : V/m               | Unit of Elec Flux Density: C/m^2
Unit of Mass             : kg                | Unit of Pressure         : Pa
Unit of Charge Density   : C/m^3             | Unit of Electric Pot     : volt
**** COMMAND INPUT *** SELECT ACTION=DEFAULT
**** COMMAND INPUT *** SELECT ACTION=SELECT OPTION=SURFACES
**** COMMAND INPUT *** THREED OPTION=REFRESH
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=-0.4 Z2=0.4 NP=100
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine' XCOMPONENT=z YCOMPONENT=Bz INTERPOLATION=LINEAR GRAPH='Default'
Integral: -1.39449e-06
Maximum: -1.64959e-08 (X at maximum: -0.08)
Minimum: -2.02736e-05 (X at minimum: -0.4)
**** COMMAND INPUT ***
**** COMMAND INPUT *** CARTESIAN  Z1=0.5 Z4=Z1 X1=-0.5 X2=X1 Z3=0.5 Z2=Z3 X3=-0.5 X4=X3 Y1=0 Y2=Y1 Y3=Y1 Y4=Y1 CORNERS=4 N1=100 N2=100 BUFFERNAME=Cartesian
**** COMMAND INPUT *** MAP FILE=TEMP COMPONENT=Bz
Component: BZ
Minimum: -1.44000939685109E-05, Maximum: -1.44000939685109E-05
Integral = 0.0
**** COMMAND INPUT ***
**** COMMAND INPUT *** CARTESIAN  Z1=0.5 Z4=Z1 X1=0.5 X2=X1 Z3=0.5 Z2=Z3 X3=-0.5 X4=X3 Y1=0 Y2=Y1 Y3=Y1 Y4=Y1 CORNERS=4 N1=100 N2=100 BUFFERNAME=Cartesian
**** COMMAND INPUT *** MAP FILE=TEMP COMPONENT=Bz
Component: BZ
Minimum: -2.02838155150378E-05, Maximum: -1.44000939685109E-05
Integral = -1.09664081597396E-19
**** COMMAND INPUT ***
**** COMMAND INPUT *** CARTESIAN  Z1=0.5 Z4=Z1 X1=0.5 X2=X1 Z3=-0.5 Z2=Z3 X3=-0.5 X4=X3 Y1=0 Y2=Y1 Y3=Y1 Y4=Y1 CORNERS=4 N1=100 N2=100 BUFFERNAME=Cartesian
**** COMMAND INPUT *** MAP FILE=TEMP COMPONENT=Bz
Component: BZ
Minimum: -2.23094292138433E-03, Maximum: 1.56878103544081E-05
Integral = -9.95752890048308E-06
**** COMMAND INPUT ***
**** COMMAND INPUT *** CARTESIAN  Z1=0.35 Z4=Z1 X1=0.35 X2=X1 Z3=-035 Z2=Z3 X3=-0.35 X4=X3 Y1=0 Y2=Y1 Y3=Y1 Y4=Y1 CORNERS=4 N1=100 N2=100 BUFFERNAME=Cartesian
Coordinates (0.35,0.0,-2.14949494949495)
Failed to evaluate fields at 10000 points.
Point outside finite element mesh.
Coordinates (0.35,0.0,-2.14949494949495)
**** COMMAND INPUT ***
**** COMMAND INPUT *** CARTESIAN  Z1=0.35 Z4=Z1 X1=0.35 X2=X1 Z3=-0.35 Z2=Z3 X3=-0.35 X4=X3 Y1=0 Y2=Y1 Y3=Y1 Y4=Y1 CORNERS=4 N1=100 N2=100 BUFFERNAME=Cartesian
**** COMMAND INPUT *** MAP FILE=TEMP COMPONENT=Bz
Component: BZ
Minimum: -4.15952364915458E-06, Maximum: 8.66452902857809E-08
Integral = -8.63189581672155E-08
**** COMMAND INPUT ***
**** COMMAND INPUT *** TABLE INFILE=TEMP  OUTFILE='C:\Users\mzhao\OPERA_miniMSR\results\mini_MSR_zaxis_Bz.table' COLUMNS=2 F1=z UNIT1T=LENGU F2=Bz UNIT2T=FLUXU UNIT3T=1 UNIT4T=1 UNIT5T=1 UNIT6T=1 UNIT7T=1 UNIT8T=1 UNIT9T=1 UNIT10T=1 UNIT11T=1 UNIT12T=1
Opening file for writing: C:\Users\mzhao\OPERA_miniMSR\results\mini_MSR_zaxis_Bz.table
**** COMMAND INPUT ***
**** COMMAND INPUT *** POINT XP=0 YP=0 ZP=0 COMPONENT=Bz
At (0.0,0.0,0.0), BZ = -1.65681958341562E-08
**** COMMAND INPUT ***
Opera-3d Post-Processor                complete on 04/Jun/2024 at 10:39:30
                                                00:09:57.96 (00:00:42.92 cp)
