Opera-3d Modeller                       started on 15/May/2024 at 15:27:27
Opera 2020 SP3 [Build 2]
(c) Copyright Dassault Systemes Ltd 1984-2019. All Rights Reserved.
Node: TUCAN-MAGPC. Processor: Intel64. System: Windows 10
Command language initialization.
Information: modeller.comi was not found in the local or home directory
*** CONTROL COMMAND ** $COMI FILE='C:\Users\mzhao\OPERA_miniMSR\test.comi' MODE=CONT
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=1 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#muthickness, Value=0.0015, DESCRIPTION='thickness of mumetal layer [m]'
Variable     Value            Expression
#muthickness 1.5E-03          0.0015
Description: thickness of mumetal layer [m]
**** FILE INPUT ******
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable     Value            Expression
#mu_DSL      500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=100, DESCRIPTION='data storage level for total potential air'
Variable     Value            Expression
#air_tot_DSL 100.0            100
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=200, DESCRIPTION='data storage level for reduced potential air'
Variable     Value            Expression
#air_red_DSL 200.0            200
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable     Value            Expression
#mu_flat_MES 0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable     Value            Expression
#mu_bent_MES 0.03             0.03
Description: maximum element size for bent mumetal
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable      Value            Expression
#mu_flat_MAbE 30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable      Value            Expression
#mu_bent_MAbE 30.0             30
Description: maximum angle between elements for bent mumetal
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Modelling
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** //// Inner Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//innerMSRstructure.STEP'
Imported material name from label: Cut_Extrude1
**** FILE INPUT ******
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_msr_structure' UNIQUENAME='inner_msr_structure'
**** FILE INPUT ******
*** COMMENT ********** /// Faces
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** // flat
*** COMMENT ********** / front
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00003
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
*** COMMENT ********** / back
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00005
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
*** COMMENT ********** / left and right
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00002
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00004
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
*** COMMENT ********** / top
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
*** COMMENT ********** / bottom
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00007
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create label
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='mumetalBH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=mumetalBH FILE='bhdata//PMTshieldB_anh.bh'
BHDATA label mumetalBH loaded with data from the file bhdata//PMTshieldB_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / match face label to bh label
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'mumetal'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=mumetal PLATETHICKNESS=#muthickness
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
End of command file.
*** CONTROL COMMAND ** $ END COMI
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** BOUNDARY UNPICK
**** COMMAND INPUT *** BOUNDARY GUIINIT
**** COMMAND INPUT ***
**** COMMAND INPUT *** BOUNDARY UNPICK
**** COMMAND INPUT ***
**** COMMAND INPUT *** MULTIPHYSICS OPTION=RESET
**** COMMAND INPUT ***
**** COMMAND INPUT *** ANALYSISDATA OPTION=SET PROGRAM=ELEKTRATR CONVTOL=1e-08 DELTAT=0.01 HX=0 HY=0 HZ=0 LINEAR=YES LOGLIST= LOGLIST+=TTime RHS=ADAPTIVE UPDATE=SIMPLE USEDEFORMEDMESH=NO
Analysis type changed to: ELEKTRATR
**** COMMAND INPUT ***
**** COMMAND INPUT *** BOUNDARY UNPICK
**** COMMAND INPUT *** BOUNDARY GUIINIT
**** COMMAND INPUT ***
**** COMMAND INPUT *** BOUNDARY UNPICK
**** COMMAND INPUT ***
**** COMMAND INPUT *** MULTIPHYSICS OPTION=RESET
**** COMMAND INPUT ***
**** COMMAND INPUT *** ANALYSISDATA OPTION=SET PROGRAM=TOSCAMAGN CONVTOL=1e-08 HX=0 HY=0 HZ=0 LINEAR=YES POTENTIALCUT=YES RHS=ADAPTIVE SCALEDRIVE=ALL USEDEFORMEDMESH=NO USEDIRECTSOLVER=NO
Analysis type changed to: TOSCAMAGN
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00004
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00004
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
