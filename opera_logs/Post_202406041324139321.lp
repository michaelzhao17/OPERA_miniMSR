Opera-3d Post-Processor                 started on 04/Jun/2024 at 13:24:13
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
**** COMMAND INPUT *** ACTIVATE FILE='C:\Users\mzhao\OPERA_miniMSR\miniMSR_layer_ffff.op3'
Opening file for checking: C:\Users\mzhao\OPERA_miniMSR\miniMSR_layer_ffff.op3
**** COMMAND INPUT *** LOAD
Attaching file as resident: C:\Users\mzhao\OPERA_miniMSR\miniMSR_layer_ffff.op3
Opening database ...OPERA_miniMSR\miniMSR_layer_ffff.op3, simulation number 1 on 04/Jun/2024 at 13:24:14
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
**** COMMAND INPUT *** POINT XP=0 YP=0 ZP=0 COMPONENT=X
At (0.0,0.0,0.0), X = 0.0
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=0.4 Z2=-0.4 NP=100
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine' XCOMPONENT=z YCOMPONENT=Bz INTERPOLATION=LINEAR GRAPH='Default'
Integral: -1.41255e-06
Maximum: 2.00834e-05 (X at maximum: -0.384)
Minimum: 1.86477e-08 (X at minimum: 0.008)
**** COMMAND INPUT ***
**** COMMAND INPUT *** TABLE INFILE=TEMP  OUTFILE='C:\Users\mzhao\OPERA_miniMSR\results\mini_MSR_ffff.table' COLUMNS=2 F1=z UNIT1T=LENGU F2=Bz UNIT2T=FLUXU UNIT3T=1 UNIT4T=1 UNIT5T=1 UNIT6T=1 UNIT7T=1 UNIT8T=1 UNIT9T=1 UNIT10T=1 UNIT11T=1 UNIT12T=1
Opening file for writing: C:\Users\mzhao\OPERA_miniMSR\results\mini_MSR_ffff.table
Do you want to overwrite "C:\Users\mzhao\OPERA_miniMSR\results\mini_MSR_ffff.table"?
YES or NO
*** LOOP COMMAND ***** yes
**** COMMAND INPUT ***
Opera-3d Post-Processor                complete on 04/Jun/2024 at 13:30:31
                                                00:06:17.53 (00:00:16.25 cp)
