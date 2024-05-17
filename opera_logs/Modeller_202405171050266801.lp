Opera-3d Modeller                       started on 17/May/2024 at 10:50:26
Opera 2020 SP3 [Build 2]
(c) Copyright Dassault Systemes Ltd 1984-2019. All Rights Reserved.
Node: TUCAN-MAGPC. Processor: Intel64. System: Windows 10
Command language initialization.
Information: modeller.comi was not found in the local or home directory
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//full_mini_MSR.STEP'
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO29#inner_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO30#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO31#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO32#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO33#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO34#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO35#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO36#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO37#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO38#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO39#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO40#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO41#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO42#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO43#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO44#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO45#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO46#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO3#outer_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO4#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO5#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO6#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO7#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO10#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO12#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO14#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO16#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO18#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO20#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO25#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO26#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO27#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO28#outer_flat_Boss_Extrude2
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
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-019' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=1 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#muthickness, Value=0.0015, DESCRIPTION='thickness of mumetal layer [m]'
Variable      Value            Expression
#muthickness  1.5E-03          0.0015
Description: thickness of mumetal layer [m]
**** FILE INPUT ******
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=100, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  100.0            100
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=200, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  200.0            200
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//full_mini_MSR.STEP'
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO29#inner_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO30#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO31#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO32#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO33#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO34#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO35#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO36#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO37#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO38#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO39#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO40#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO41#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO42#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO43#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO44#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO45#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO46#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO3#outer_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO4#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO5#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO6#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO7#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO10#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO12#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO14#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO16#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO18#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO20#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO25#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO26#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO27#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO28#outer_flat_Boss_Extrude2
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
**** COMMAND INPUT *** WINDOW OUTLINE=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW OUTLINE=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW SOLID=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW OUTLINE=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW OUTLINE=YES
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=1 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#muthickness, Value=0.0015, DESCRIPTION='thickness of mumetal layer [m]'
Variable      Value            Expression
#muthickness  1.5E-03          0.0015
Description: thickness of mumetal layer [m]
**** FILE INPUT ******
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=100, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  100.0            100
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=200, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  200.0            200
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//full_mini_MSR.STEP'
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO29#inner_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO30#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO31#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO32#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO33#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO34#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO35#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO36#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO37#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO38#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO39#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO40#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO41#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO42#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO43#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO44#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO45#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO46#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO3#outer_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO4#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO5#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO6#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO7#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO10#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO12#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO14#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO16#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO18#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO20#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO25#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO26#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO27#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO28#outer_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_layer' UNIQUENAME='inner_layer'
**** FILE INPUT ******
*** COMMENT ********** /// Faces
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** // flat
*** COMMENT ********** / front
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00003
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / back
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00005
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / left and right
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00002
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00004
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / top
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / bottom
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00007
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
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
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=1 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#muthickness, Value=0.0015, DESCRIPTION='thickness of mumetal layer [m]'
Variable      Value            Expression
#muthickness  1.5E-03          0.0015
Description: thickness of mumetal layer [m]
**** FILE INPUT ******
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=100, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  100.0            100
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=200, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  200.0            200
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//full_mini_MSR.STEP'
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO29#inner_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO30#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO31#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO32#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO33#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO34#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO35#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO36#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO37#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO38#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO39#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO40#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO41#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO42#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO43#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO44#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO45#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO46#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO3#outer_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO4#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO5#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO6#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO7#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO10#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO12#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO14#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO16#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO18#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO20#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO25#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO26#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO27#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO28#outer_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_layer' UNIQUENAME='inner_layer'
**** FILE INPUT ******
*** COMMENT ********** /// Faces
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** // flat
*** COMMENT ********** / front
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00003
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / back
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00005
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / left and right
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00002
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00004
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / top
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / bottom
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00007
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
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
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0.5 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#muthickness, Value=0.0015, DESCRIPTION='thickness of mumetal layer [m]'
Variable      Value            Expression
#muthickness  1.5E-03          0.0015
Description: thickness of mumetal layer [m]
**** FILE INPUT ******
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=100, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  100.0            100
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=200, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  200.0            200
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//full_mini_MSR.STEP'
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO29#inner_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO30#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO31#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO32#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO33#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO34#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO35#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO36#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO37#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO38#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO39#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO40#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO41#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO42#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO43#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO44#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO45#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO46#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO3#outer_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO4#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO5#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO6#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO7#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO10#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO12#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO14#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO16#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO18#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO20#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO25#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO26#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO27#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO28#outer_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_layer' UNIQUENAME='inner_layer'
**** FILE INPUT ******
*** COMMENT ********** /// Faces
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** // flat
*** COMMENT ********** / front
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00003
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / back
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00005
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / left and right
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00002
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00004
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / top
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / bottom
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00007
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
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
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0.1 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#muthickness, Value=0.0015, DESCRIPTION='thickness of mumetal layer [m]'
Variable      Value            Expression
#muthickness  1.5E-03          0.0015
Description: thickness of mumetal layer [m]
**** FILE INPUT ******
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=100, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  100.0            100
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=200, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  200.0            200
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//full_mini_MSR.STEP'
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO29#inner_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO30#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO31#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO32#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO33#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO34#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO35#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO36#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO37#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO38#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO39#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO40#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO41#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO42#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO43#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO44#inner_bracket_Cut_Extrude3
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO45#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO1#inner_layer_NAUO46#inner_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO3#outer_flat_front_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO4#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO5#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO6#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO7#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO10#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO12#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO14#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO16#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO18#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO20#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO25#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO26#outer_bracket_Cut_Extrude4
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO27#outer_flat_Boss_Extrude2
Imported material name from label: full_mini_MSR_NAUO2#outer_layer_NAUO28#outer_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_layer' UNIQUENAME='inner_layer'
**** FILE INPUT ******
*** COMMENT ********** /// Faces
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** // flat
*** COMMENT ********** / front
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00003
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / back
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00005
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / left and right
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00002
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00004
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / top
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / bottom
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00007
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
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
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=1 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#muthickness, Value=0.0015, DESCRIPTION='thickness of mumetal layer [m]'
Variable      Value            Expression
#muthickness  1.5E-03          0.0015
Description: thickness of mumetal layer [m]
**** FILE INPUT ******
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=100, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  100.0            100
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=200, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  200.0            200
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_layer' UNIQUENAME='inner_layer'
**** FILE INPUT ******
*** COMMENT ********** /// Faces
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** // flat
*** COMMENT ********** / front
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00003
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / back
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00005
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / left and right
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00002
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00004
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / top
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / bottom
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00007
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
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
**** COMMAND INPUT *** THREED OPTION=SETVIEW SIZE=0
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#muthickness, Value=0.0015, DESCRIPTION='thickness of mumetal layer [m]'
Variable      Value            Expression
#muthickness  1.5E-03          0.0015
Description: thickness of mumetal layer [m]
**** FILE INPUT ******
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=100, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  100.0            100
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=200, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  200.0            200
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_layer' UNIQUENAME='inner_layer'
**** FILE INPUT ******
*** COMMENT ********** /// Faces
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** // flat
*** COMMENT ********** / front
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00003
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / back
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00005
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / left and right
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00002
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00004
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / top
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / bottom
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00007
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
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
**** COMMAND INPUT *** WINDOW SOLID=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW SOLID=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER COMMAND=CURSORHIDE
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=FACE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-017' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-017' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-017' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-017' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT *** UNDO
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=BODY
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER COMMAND=PICK
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_layer'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_layer'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=FACE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_layer' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-018' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-010' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-010' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-018' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-017' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_layer' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-007' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-017' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-004' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-004' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-004' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-017' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-010' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-017' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-004' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_layer' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#muthickness, Value=0.0015, DESCRIPTION='thickness of mumetal layer [m]'
Variable      Value            Expression
#muthickness  1.5E-03          0.0015
Description: thickness of mumetal layer [m]
**** FILE INPUT ******
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=100, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  100.0            100
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=200, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  200.0            200
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_layer' UNIQUENAME='inner_layer'
**** FILE INPUT ******
*** COMMENT ********** /// Faces
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** // flat
*** COMMENT ********** / front
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00007
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / back
**** FILE INPUT ******
*** COMMENT ********** / left and right
**** FILE INPUT ******
*** COMMENT ********** / top
**** FILE INPUT ******
*** COMMENT ********** / bottom
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
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
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_layer' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=FACE
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=FACE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_layer' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-010' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-004' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-004' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-010' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_layer' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-004' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-010' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-011' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-005' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-017' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW SOLID=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW OUTLINE=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW OUTLINE=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW SOLID=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_layer' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=BODY
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_layer'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#muthickness, Value=0.0015, DESCRIPTION='thickness of mumetal layer [m]'
Variable      Value            Expression
#muthickness  1.5E-03          0.0015
Description: thickness of mumetal layer [m]
**** FILE INPUT ******
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=100, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  100.0            100
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=200, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  200.0            200
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** /// Faces
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** // flat
*** COMMENT ********** / front
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_msr_structure' IDENTIFIER=A.00007
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
At least one face must be picked
*** COMMENT ********** / back
**** FILE INPUT ******
*** COMMENT ********** / left and right
**** FILE INPUT ******
*** COMMENT ********** / top
**** FILE INPUT ******
*** COMMENT ********** / bottom
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
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
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=FACE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-018' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER COMMAND=CURSORHIDE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-018' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-018' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-018' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-018' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER COMMAND=PICK
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00006
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=BODY
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=1 SIZE=2 NORMALTOL=3 ELEMSHAPEPREF=NONE
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=FACE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00006
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_inner_thickness, Value=0.0015, DESCRIPTION='thickness of mumetal inner layer [m]'
Variable            Value            Expression
#mu_inner_thickness 1.5E-03          0.0015
Description: thickness of mumetal inner layer [m]
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_outer_thickness, Value=0.002, DESCRIPTION='thickness of mumetal outer layer [m]'
Variable            Value            Expression
#mu_outer_thickness 2.0E-03          0.002
Description: thickness of mumetal outer layer [m]
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable            Value            Expression
#mu_DSL             500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable            Value            Expression
#air_tot_DSL        200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable            Value            Expression
#air_red_DSL        100.0            100
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable            Value            Expression
#mu_flat_MES        0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable            Value            Expression
#mu_bent_MES        0.03             0.03
Description: maximum element size for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_bent_MES, Value=0.3, DESCRIPTION='maximum element size for air'
Variable            Value            Expression
#air_bent_MES       0.3              0.3
Description: maximum element size for air
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable            Value            Expression
#mu_flat_MAbE       30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable            Value            Expression
#mu_bent_MAbE       30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable            Value            Expression
#air_bent_MAbE      30.0             30
Description: maximum angle between elements for air
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** // Front panel
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=UNPICK REDISPLAY=NO
*** COMMENT ********** / set material to air
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_flat_MES NORMALTOL=#air_flat_MAbE ELEMSHAPEPREF=NONE
Undefined variable: #AIR_FLAT_MES
DCOD Message  74:Unknown variable in expression '#air_flat_MES' (DECODE)
Undefined variable: #AIR_FLAT_MABE
DCOD Message  74:Unknown variable in expression '#air_flat_MAbE' (DECODE)
Command file processing aborted
Check the log file or console for details.
Error in file: 'C:\Users\mzhao\OPERA_miniMSR\test.comi' (Line 60)
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_inner_thickness, Value=0.0015, DESCRIPTION='thickness of mumetal inner layer [m]'
Variable            Value            Expression
#mu_inner_thickness 1.5E-03          0.0015
Description: thickness of mumetal inner layer [m]
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_outer_thickness, Value=0.002, DESCRIPTION='thickness of mumetal outer layer [m]'
Variable            Value            Expression
#mu_outer_thickness 2.0E-03          0.002
Description: thickness of mumetal outer layer [m]
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable            Value            Expression
#mu_DSL             500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable            Value            Expression
#air_tot_DSL        200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable            Value            Expression
#air_red_DSL        100.0            100
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable            Value            Expression
#mu_flat_MES        0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable            Value            Expression
#mu_bent_MES        0.03             0.03
Description: maximum element size for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MES, Value=0.3, DESCRIPTION='maximum element size for air'
Variable            Value            Expression
#air_MES            0.3              0.3
Description: maximum element size for air
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable            Value            Expression
#mu_flat_MAbE       30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable            Value            Expression
#mu_bent_MAbE       30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable            Value            Expression
#air_MAbE           30.0             30
Description: maximum angle between elements for air
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** // Front panel
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=UNPICK REDISPLAY=NO
*** COMMENT ********** / set material to air
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
At least one cell or body must be picked
**** FILE INPUT ******
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
*** COMMENT ********** / back
**** FILE INPUT ******
*** COMMENT ********** / left and right
**** FILE INPUT ******
*** COMMENT ********** / top
**** FILE INPUT ******
*** COMMENT ********** / bottom
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
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
**** COMMAND INPUT *** FILTER COMMAND=CURSORHIDE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-018' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-018' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-018' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-018' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER COMMAND=PICK
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=500 ELEMENTTYPE=Linear SIZE=0.05 NORMALTOL=30 FORMETHOD=NONE BACKMETHOD=NONE
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=BODY
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=3 SIZE=2 NORMALTOL=1 ELEMSHAPEPREF=NONE
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_inner_thickness, Value=0.0015, DESCRIPTION='thickness of mumetal inner layer [m]'
Variable            Value            Expression
#mu_inner_thickness 1.5E-03          0.0015
Description: thickness of mumetal inner layer [m]
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_outer_thickness, Value=0.002, DESCRIPTION='thickness of mumetal outer layer [m]'
Variable            Value            Expression
#mu_outer_thickness 2.0E-03          0.002
Description: thickness of mumetal outer layer [m]
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable            Value            Expression
#mu_DSL             500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable            Value            Expression
#air_tot_DSL        200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable            Value            Expression
#air_red_DSL        100.0            100
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable            Value            Expression
#mu_flat_MES        0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable            Value            Expression
#mu_bent_MES        0.03             0.03
Description: maximum element size for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MES, Value=0.3, DESCRIPTION='maximum element size for air'
Variable            Value            Expression
#air_MES            0.3              0.3
Description: maximum element size for air
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable            Value            Expression
#mu_flat_MAbE       30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable            Value            Expression
#mu_bent_MAbE       30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable            Value            Expression
#air_MAbE           30.0             30
Description: maximum angle between elements for air
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** // Front panel
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
At least one cell or body must be picked
**** FILE INPUT ****** PREVIEW OPTION=UNPICK REDISPLAY=NO
**** FILE INPUT ******
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
*** COMMENT ********** / back
**** FILE INPUT ******
*** COMMENT ********** / left and right
**** FILE INPUT ******
*** COMMENT ********** / top
**** FILE INPUT ******
*** COMMENT ********** / bottom
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
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
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=BODY
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_inner_thickness, Value=0.0015, DESCRIPTION='thickness of mumetal inner layer [m]'
Variable            Value            Expression
#mu_inner_thickness 1.5E-03          0.0015
Description: thickness of mumetal inner layer [m]
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_outer_thickness, Value=0.002, DESCRIPTION='thickness of mumetal outer layer [m]'
Variable            Value            Expression
#mu_outer_thickness 2.0E-03          0.002
Description: thickness of mumetal outer layer [m]
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable            Value            Expression
#mu_DSL             500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable            Value            Expression
#air_tot_DSL        200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable            Value            Expression
#air_red_DSL        100.0            100
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable            Value            Expression
#mu_flat_MES        0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable            Value            Expression
#mu_bent_MES        0.03             0.03
Description: maximum element size for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MES, Value=0.3, DESCRIPTION='maximum element size for air'
Variable            Value            Expression
#air_MES            0.3              0.3
Description: maximum element size for air
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable            Value            Expression
#mu_flat_MAbE       30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable            Value            Expression
#mu_bent_MAbE       30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable            Value            Expression
#air_MAbE           30.0             30
Description: maximum angle between elements for air
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** // Front panel
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
At least one cell or body must be picked
*** COMMENT ********** / set material to air
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=UNPICK REDISPLAY=NO
**** FILE INPUT ******
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
*** COMMENT ********** / back
**** FILE INPUT ******
*** COMMENT ********** / left and right
**** FILE INPUT ******
*** COMMENT ********** / top
**** FILE INPUT ******
*** COMMENT ********** / bottom
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
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
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=BODY
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Default ELEMENTTYPE=Linear ELEMSHAPEPREF=NONE
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_inner_thickness, Value=0.0015, DESCRIPTION='thickness of mumetal inner layer [m]'
Variable            Value            Expression
#mu_inner_thickness 1.5E-03          0.0015
Description: thickness of mumetal inner layer [m]
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_outer_thickness, Value=0.002, DESCRIPTION='thickness of mumetal outer layer [m]'
Variable            Value            Expression
#mu_outer_thickness 2.0E-03          0.002
Description: thickness of mumetal outer layer [m]
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable            Value            Expression
#mu_DSL             500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable            Value            Expression
#air_tot_DSL        200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable            Value            Expression
#air_red_DSL        100.0            100
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable            Value            Expression
#mu_flat_MES        0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable            Value            Expression
#mu_bent_MES        0.03             0.03
Description: maximum element size for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MES, Value=0.3, DESCRIPTION='maximum element size for air'
Variable            Value            Expression
#air_MES            0.3              0.3
Description: maximum element size for air
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable            Value            Expression
#mu_flat_MAbE       30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable            Value            Expression
#mu_bent_MAbE       30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable            Value            Expression
#air_MAbE           30.0             30
Description: maximum angle between elements for air
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** // Front panel
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=UNPICK REDISPLAY=NO
**** FILE INPUT ******
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
*** COMMENT ********** / back
**** FILE INPUT ******
*** COMMENT ********** / left and right
**** FILE INPUT ******
*** COMMENT ********** / top
**** FILE INPUT ******
*** COMMENT ********** / bottom
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
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
**** COMMAND INPUT *** FILTER TYPE=BODY
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=-65 ROTY=0 ROTZ=-45 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** / thickness of mumetal layer
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_inner_thickness, Value=0.0015, DESCRIPTION='thickness of mumetal inner layer [m]'
Variable            Value            Expression
#mu_inner_thickness 1.5E-03          0.0015
Description: thickness of mumetal inner layer [m]
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_outer_thickness, Value=0.002, DESCRIPTION='thickness of mumetal outer layer [m]'
Variable            Value            Expression
#mu_outer_thickness 2.0E-03          0.002
Description: thickness of mumetal outer layer [m]
*** COMMENT ********** // data storage levels (DSL)
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable            Value            Expression
#mu_DSL             500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable            Value            Expression
#air_tot_DSL        200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable            Value            Expression
#air_red_DSL        100.0            100
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable            Value            Expression
#mu_flat_MES        0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable            Value            Expression
#mu_bent_MES        0.03             0.03
Description: maximum element size for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MES, Value=0.3, DESCRIPTION='maximum element size for air'
Variable            Value            Expression
#air_MES            0.3              0.3
Description: maximum element size for air
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable            Value            Expression
#mu_flat_MAbE       30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable            Value            Expression
#mu_bent_MAbE       30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable            Value            Expression
#air_MAbE           30.0             30
Description: maximum angle between elements for air
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
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** // Front panel
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
*** COMMENT ********** / back
**** FILE INPUT ******
*** COMMENT ********** / left and right
**** FILE INPUT ******
*** COMMENT ********** / top
**** FILE INPUT ******
*** COMMENT ********** / bottom
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** / brackets
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
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
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00007
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=BODY
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
Opera-3d Modeller                      complete on 17/May/2024 at 12:10:16
                                                01:19:50.59 (00:04:59.53 cp)
