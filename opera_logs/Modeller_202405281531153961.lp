Opera-3d Modeller                       started on 28/May/2024 at 15:31:15
Opera 2021 SP3 [Build 3]
(c) Copyright Dassault Systemes Ltd 1984-2021. All Rights Reserved.
Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009 windows (10.0.19045)
Command language initialization.
Information: modeller.comi was not found in the local or home directory
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA with layered meshing
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** //Data Storage Levels
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable Value            Expression
#mu_DSL  500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable     Value            Expression
#air_tot_DSL 200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable     Value            Expression
#air_red_DSL 100.0            100
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
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.02, DESCRIPTION='maximum element size for total potential air'
Variable     Value            Expression
#air_tot_MES 0.02             0.02
Description: maximum element size for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_MES, Value=0.3, DESCRIPTION='maximum element size for reduced potential air'
Variable     Value            Expression
#air_red_MES 0.3              0.3
Description: maximum element size for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable      Value            Expression
#mu_flat_MAbE 30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable      Value            Expression
#mu_bent_MAbE 30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable      Value            Expression
#air_MAbE     30.0             30
Description: maximum angle between elements for air
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
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_left' UNIQUENAME='inner_brkt_bottom_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_door_BH FILE='bhdata//miniMSR_inner_door_anh.bh'
BHDATA label innermumetal_door_BH loaded with data from the file bhdata//miniMSR_inner_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_floor_BH FILE='bhdata//miniMSR_inner_floor_anh.bh'
BHDATA label innermumetal_floor_BH loaded with data from the file bhdata//miniMSR_inner_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Outer Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//outer_layer.STEP'
Imported material name from label: outer_layer_NAUO1#outer_flat_front_Boss_Extrude2
Imported material name from label: outer_layer_NAUO2#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO3#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO4#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO5#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO6#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO7#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO10#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO12#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO14#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO16#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO18#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO20#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO25#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO26#outer_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_front' UNIQUENAME='outer_front'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-026'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_back' UNIQUENAME='outer_back'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_left' UNIQUENAME='outer_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_right' UNIQUENAME='outer_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-025'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_top' UNIQUENAME='outer_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_bottom' UNIQUENAME='outer_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-021'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_top' UNIQUENAME='outer_brkt_front_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_bottom' UNIQUENAME='outer_brkt_front_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_left' UNIQUENAME='outer_brkt_front_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_right' UNIQUENAME='outer_brkt_front_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom_left' UNIQUENAME='outer_brkt_bottom_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-019'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom_right' UNIQUENAME='outer_brkt_bottom_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-022'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_left' UNIQUENAME='outer_brkt_top_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-024'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_right' UNIQUENAME='outer_brkt_top_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-020'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_bottom' UNIQUENAME='outer_brkt_back_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-023'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_top' UNIQUENAME='outer_brkt_back_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_left' UNIQUENAME='outer_brkt_back_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_right' UNIQUENAME='outer_brkt_back_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_door_BH FILE='bhdata//miniMSR_outer_door_anh.bh'
BHDATA label outermumetal_door_BH loaded with data from the file bhdata//miniMSR_outer_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_floor_BH FILE='bhdata//miniMSR_outer_floor_anh.bh'
BHDATA label outermumetal_floor_BH loaded with data from the file bhdata//miniMSR_outer_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
**** FILE INPUT ****** BACKGROUND OPTION=LOAD
**** FILE INPUT ****** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=5 SCALEY=5 SCALEZ=5   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** //// Meshing
*** COMMENT ********** / Surface Mesh
**** FILE INPUT ****** MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
1106 faces completed, generating 105628 surface elements

Triangular facet quality on 1106 faces
Triangles: 105628, Max: 1.0, Min: 0.0434594, Mean: 0.741535, Power2Mean: 0.168043

Face 2 is marked 'Surface mesh warning'
Face 127 is marked 'Surface mesh warning'
Face 132 is marked 'Surface mesh warning'
Face 135 is marked 'Surface mesh warning'
Face 165 is marked 'Surface mesh warning'
Face 170 is marked 'Surface mesh warning'
Face 181 is marked 'Surface mesh warning'
Face 196 is marked 'Surface mesh warning'
Face 221 is marked 'Surface mesh warning'
Face 225 is marked 'Surface mesh warning'
Face 236 is marked 'Surface mesh warning'
Face 338 is marked 'Surface mesh warning'
Face 354 is marked 'Surface mesh warning'
Face 357 is marked 'Surface mesh warning'
Face 367 is marked 'Surface mesh warning'
Face 379 is marked 'Surface mesh warning'
Face 397 is marked 'Surface mesh warning'
Face 405 is marked 'Surface mesh warning'
Face 408 is marked 'Surface mesh warning'
Face 564 is marked 'Surface mesh warning'
Face 666 is marked 'Surface mesh warning'
Face 768 is marked 'Surface mesh warning'
Face 922 is marked 'Surface mesh warning'
Face 934 is marked 'Surface mesh warning'
Face 950 is marked 'Surface mesh warning'
Face 960 is marked 'Surface mesh warning'
Face 971 is marked 'Surface mesh warning'
Checking triangular mesh on 1106 faces issued warning for 27 faces
Checking triangular mesh on faces issued warning
You can go forward, while you are advised to check flagged faces.
**** FILE INPUT ******
*** COMMENT ********** / Volume Mesh
**** FILE INPUT ****** FILL TOL=1.0E-06
Mesh Generator Type II Error Number : 5
 Error Message : Failed to generate tet mesh: Call to VKI function returned invalid data
vis_TetMeshGenerate : SYS_ERROR_OPERATION   An operation failed in current state : Input face intersection
Command file processing aborted
Check the log file or console for details.
Error in file: 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi' (Line 681)
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-05
Mesh Generator Type II Error Number : 5
 Error Message : Failed to generate tet mesh: Call to VKI function returned invalid data
vis_TetMeshGenerate : SYS_ERROR_OPERATION   An operation failed in current state : Input face intersection
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA with layered meshing
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
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** //Data Storage Levels
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  100.0            100
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
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.02, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.02             0.02
Description: maximum element size for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_MES, Value=0.3, DESCRIPTION='maximum element size for reduced potential air'
Variable      Value            Expression
#air_red_MES  0.3              0.3
Description: maximum element size for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable      Value            Expression
#mu_flat_MAbE 30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable      Value            Expression
#mu_bent_MAbE 30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable      Value            Expression
#air_MAbE     30.0             30
Description: maximum angle between elements for air
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
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_left' UNIQUENAME='inner_brkt_bottom_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_door_BH FILE='bhdata//miniMSR_inner_door_anh.bh'
BHDATA label innermumetal_door_BH loaded with data from the file bhdata//miniMSR_inner_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_floor_BH FILE='bhdata//miniMSR_inner_floor_anh.bh'
BHDATA label innermumetal_floor_BH loaded with data from the file bhdata//miniMSR_inner_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Outer Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//outer_layer.STEP'
Imported material name from label: outer_layer_NAUO1#outer_flat_front_Boss_Extrude2
Imported material name from label: outer_layer_NAUO2#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO3#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO4#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO5#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO6#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO7#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO10#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO12#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO14#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO16#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO18#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO20#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO25#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO26#outer_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_front' UNIQUENAME='outer_front'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-026'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_back' UNIQUENAME='outer_back'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_left' UNIQUENAME='outer_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_right' UNIQUENAME='outer_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-025'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_top' UNIQUENAME='outer_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_bottom' UNIQUENAME='outer_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-021'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_top' UNIQUENAME='outer_brkt_front_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_bottom' UNIQUENAME='outer_brkt_front_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_left' UNIQUENAME='outer_brkt_front_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_right' UNIQUENAME='outer_brkt_front_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom_left' UNIQUENAME='outer_brkt_bottom_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-019'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom_right' UNIQUENAME='outer_brkt_bottom_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-022'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_left' UNIQUENAME='outer_brkt_top_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-024'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_right' UNIQUENAME='outer_brkt_top_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-020'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_bottom' UNIQUENAME='outer_brkt_back_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-023'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_top' UNIQUENAME='outer_brkt_back_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_left' UNIQUENAME='outer_brkt_back_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_right' UNIQUENAME='outer_brkt_back_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_door_BH FILE='bhdata//miniMSR_outer_door_anh.bh'
BHDATA label outermumetal_door_BH loaded with data from the file bhdata//miniMSR_outer_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_floor_BH FILE='bhdata//miniMSR_outer_floor_anh.bh'
BHDATA label outermumetal_floor_BH loaded with data from the file bhdata//miniMSR_outer_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
**** FILE INPUT ****** BACKGROUND OPTION=LOAD
**** FILE INPUT ****** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=5 SCALEY=5 SCALEZ=5   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** //// Meshing
*** COMMENT ********** / Surface Mesh
**** FILE INPUT ****** MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
1036 faces completed, generating 97862 surface elements

Triangular facet quality on 1036 faces
Triangles: 97862, Max: 1.0, Min: 0.0461084, Mean: 0.75027, Power2Mean: 0.186536

Face 2 is marked 'Surface mesh warning'
Checking triangular mesh on 1036 faces issued warning for a face
Checking triangular mesh on faces issued warning
You can go forward, while you are advised to check flagged faces.
**** FILE INPUT ******
*** COMMENT ********** / Volume Mesh
**** FILE INPUT ****** FILL TOL=1.0E-05
Mesh Generator Type II Error Number : 5
 Error Message : Failed to generate tet mesh: Call to VKI function returned invalid data
vis_TetMeshGenerate : SYS_ERROR_OPERATION   An operation failed in current state : Input face intersection
Command file processing aborted
Check the log file or console for details.
Error in file: 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi' (Line 681)
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA with layered meshing
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
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** //Data Storage Levels
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  100.0            100
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
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.02, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.02             0.02
Description: maximum element size for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_MES, Value=0.3, DESCRIPTION='maximum element size for reduced potential air'
Variable      Value            Expression
#air_red_MES  0.3              0.3
Description: maximum element size for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable      Value            Expression
#mu_flat_MAbE 30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable      Value            Expression
#mu_bent_MAbE 30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable      Value            Expression
#air_MAbE     30.0             30
Description: maximum angle between elements for air
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
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_left' UNIQUENAME='inner_brkt_bottom_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0003
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_door_BH FILE='bhdata//miniMSR_inner_door_anh.bh'
BHDATA label innermumetal_door_BH loaded with data from the file bhdata//miniMSR_inner_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_floor_BH FILE='bhdata//miniMSR_inner_floor_anh.bh'
BHDATA label innermumetal_floor_BH loaded with data from the file bhdata//miniMSR_inner_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Outer Layer
**** FILE INPUT ******
*** COMMENT ********** /++ / load structural step file
*** COMMENT ********** /++ LOAD OPTION=INSERT FILE='stpfiles//outer_layer.STEP'
*** COMMENT ********** /++
*** COMMENT ********** /++ /// Flat panels
*** COMMENT ********** /++ // front
*** COMMENT ********** /++ / rename body
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-001'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_front' UNIQUENAME='outer_front'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_front'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_front' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // back
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-026'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_back' UNIQUENAME='outer_back'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_back' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-014'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_left' UNIQUENAME='outer_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_left' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-018'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_right' UNIQUENAME='outer_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_right' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-025'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_top' UNIQUENAME='outer_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-005'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_bottom' UNIQUENAME='outer_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ /// brackets
*** COMMENT ********** /++ // front-top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-021'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_top' UNIQUENAME='outer_brkt_front_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-003'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_bottom' UNIQUENAME='outer_brkt_front_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-013'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_left' UNIQUENAME='outer_brkt_front_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-017'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_right' UNIQUENAME='outer_brkt_front_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-015'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_bottom_left' UNIQUENAME='outer_brkt_bottom_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-019'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_bottom_right' UNIQUENAME='outer_brkt_bottom_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // top-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-022'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_top_left' UNIQUENAME='outer_brkt_top_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // top-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-024'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_top_right' UNIQUENAME='outer_brkt_top_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-020'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_bottom' UNIQUENAME='outer_brkt_back_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-023'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_top' UNIQUENAME='outer_brkt_back_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-012'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_left' UNIQUENAME='outer_brkt_back_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-016'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_right' UNIQUENAME='outer_brkt_back_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // BH data
*** COMMENT ********** /++ / create labels
*** COMMENT ********** /++ BHDATA OPTION=NEW  LABEL='outermumetal_door_BH'
*** COMMENT ********** /++ BHDATA OPTION=NEW  LABEL='outermumetal_floor_BH'
*** COMMENT ********** /++
*** COMMENT ********** /++ / assign bh file to label
*** COMMENT ********** /++ BHDATA OPTION=LOAD LABEL=outermumetal_door_BH FILE='bhdata//miniMSR_outer_door_anh.bh'
*** COMMENT ********** /++ BHDATA OPTION=LOAD LABEL=outermumetal_floor_BH FILE='bhdata//miniMSR_outer_floor_anh.bh'
*** COMMENT ********** /++
*** COMMENT ********** /++ / match material label to bh label
*** COMMENT ********** /++ MATERIALS UNPICK
*** COMMENT ********** / MATERIALS GUIINIT
*** COMMENT ********** /++ MATERIALS PICK 'outer_mumetal_door'
*** COMMENT ********** /++ MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_door_BH'
*** COMMENT ********** / MATERIALS UNPICK
*** COMMENT ********** /++ MATERIALS UNPICK
*** COMMENT ********** / MATERIALS GUIINIT
*** COMMENT ********** /++ MATERIALS PICK 'outer_mumetal_floor'
*** COMMENT ********** /++ MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_floor_BH'
*** COMMENT ********** / MATERIALS UNPICK
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
**** FILE INPUT ****** BACKGROUND OPTION=LOAD
**** FILE INPUT ****** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=5 SCALEY=5 SCALEZ=5   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** //// Meshing
*** COMMENT ********** / Surface Mesh
**** FILE INPUT ****** MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
521 faces completed, generating 48848 surface elements

Triangular facet quality on 521 faces
Triangles: 48848, Max: 1.0, Min: 0.0445743, Mean: 0.760972, Power2Mean: 0.169742

Face 1 is marked 'Surface mesh warning'
Checking triangular mesh on 521 faces issued warning for a face
Checking triangular mesh on faces issued warning
You can go forward, while you are advised to check flagged faces.
**** FILE INPUT ******
*** COMMENT ********** / Volume Mesh
**** FILE INPUT ****** FILL TOL=1.0E-05
Mesh Generator Type II Error Number : 5
 Error Message : Failed to generate tet mesh: Call to VKI function returned invalid data
vis_TetMeshGenerate : SYS_ERROR_OPERATION   An operation failed in current state : Input face intersection
Command file processing aborted
Check the log file or console for details.
Error in file: 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi' (Line 681)
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL DELETE
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER COMMAND=CURSORHIDE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA with layered meshing
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
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** //Data Storage Levels
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  100.0            100
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
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.02, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.02             0.02
Description: maximum element size for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_MES, Value=0.3, DESCRIPTION='maximum element size for reduced potential air'
Variable      Value            Expression
#air_red_MES  0.3              0.3
Description: maximum element size for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable      Value            Expression
#mu_flat_MAbE 30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable      Value            Expression
#mu_bent_MAbE 30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable      Value            Expression
#air_MAbE     30.0             30
Description: maximum angle between elements for air
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
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_left' UNIQUENAME='inner_brkt_bottom_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_door_BH FILE='bhdata//miniMSR_inner_door_anh.bh'
BHDATA label innermumetal_door_BH loaded with data from the file bhdata//miniMSR_inner_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_floor_BH FILE='bhdata//miniMSR_inner_floor_anh.bh'
BHDATA label innermumetal_floor_BH loaded with data from the file bhdata//miniMSR_inner_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Outer Layer
**** FILE INPUT ******
*** COMMENT ********** /++ / load structural step file
*** COMMENT ********** /++ LOAD OPTION=INSERT FILE='stpfiles//outer_layer.STEP'
*** COMMENT ********** /++
*** COMMENT ********** /++ /// Flat panels
*** COMMENT ********** /++ // front
*** COMMENT ********** /++ / rename body
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-001'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_front' UNIQUENAME='outer_front'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_front'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_front' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // back
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-026'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_back' UNIQUENAME='outer_back'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_back' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-014'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_left' UNIQUENAME='outer_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_left' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-018'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_right' UNIQUENAME='outer_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_right' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-025'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_top' UNIQUENAME='outer_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-005'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_bottom' UNIQUENAME='outer_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ /// brackets
*** COMMENT ********** /++ // front-top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-021'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_top' UNIQUENAME='outer_brkt_front_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-003'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_bottom' UNIQUENAME='outer_brkt_front_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-013'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_left' UNIQUENAME='outer_brkt_front_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-017'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_right' UNIQUENAME='outer_brkt_front_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-015'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_bottom_left' UNIQUENAME='outer_brkt_bottom_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-019'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_bottom_right' UNIQUENAME='outer_brkt_bottom_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // top-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-022'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_top_left' UNIQUENAME='outer_brkt_top_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // top-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-024'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_top_right' UNIQUENAME='outer_brkt_top_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-020'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_bottom' UNIQUENAME='outer_brkt_back_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-023'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_top' UNIQUENAME='outer_brkt_back_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-012'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_left' UNIQUENAME='outer_brkt_back_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-016'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_right' UNIQUENAME='outer_brkt_back_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // BH data
*** COMMENT ********** /++ / create labels
*** COMMENT ********** /++ BHDATA OPTION=NEW  LABEL='outermumetal_door_BH'
*** COMMENT ********** /++ BHDATA OPTION=NEW  LABEL='outermumetal_floor_BH'
*** COMMENT ********** /++
*** COMMENT ********** /++ / assign bh file to label
*** COMMENT ********** /++ BHDATA OPTION=LOAD LABEL=outermumetal_door_BH FILE='bhdata//miniMSR_outer_door_anh.bh'
*** COMMENT ********** /++ BHDATA OPTION=LOAD LABEL=outermumetal_floor_BH FILE='bhdata//miniMSR_outer_floor_anh.bh'
*** COMMENT ********** /++
*** COMMENT ********** /++ / match material label to bh label
*** COMMENT ********** /++ MATERIALS UNPICK
*** COMMENT ********** / MATERIALS GUIINIT
*** COMMENT ********** /++ MATERIALS PICK 'outer_mumetal_door'
*** COMMENT ********** /++ MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_door_BH'
*** COMMENT ********** / MATERIALS UNPICK
*** COMMENT ********** /++ MATERIALS UNPICK
*** COMMENT ********** / MATERIALS GUIINIT
*** COMMENT ********** /++ MATERIALS PICK 'outer_mumetal_floor'
*** COMMENT ********** /++ MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_floor_BH'
*** COMMENT ********** / MATERIALS UNPICK
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
**** FILE INPUT ****** BACKGROUND OPTION=LOAD
**** FILE INPUT ****** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=5 SCALEY=5 SCALEZ=5   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** //// Meshing
*** COMMENT ********** / Surface Mesh
**** FILE INPUT ****** MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
271 faces completed, generating 25211 surface elements

Triangular facet quality on 271 faces
Triangles: 25211, Max: 1.0, Min: 0.081491, Mean: 0.830982, Power2Mean: 0.367169

**** FILE INPUT ******
*** COMMENT ********** / Volume Mesh
**** FILE INPUT ****** FILL TOL=1.0E-05
Mesh Generator Type II Error Number : 5
 Error Message : Failed to generate tet mesh: Call to VKI function returned invalid data
vis_TetMeshGenerate : SYS_ERROR_OPERATION   An operation failed in current state : Input face intersection
Command file processing aborted
Check the log file or console for details.
Error in file: 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi' (Line 681)
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
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
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** WCS OPTION=PICKED
Updating LCS name to 'Local coordinate system:AAB-001'
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PICK OPTION=TOGGLE TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PICK OPTION=TOGGLE TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** PICK OPTION=TOGGLE TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
**** COMMAND INPUT *** PICK OPTION=TOGGLE TYPE=FACE UNIQUEBODYNAME='ModelBody' IDENTIFIER=BL.00004
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA with layered meshing
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
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** //Data Storage Levels
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  100.0            100
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
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.02, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.02             0.02
Description: maximum element size for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_MES, Value=0.3, DESCRIPTION='maximum element size for reduced potential air'
Variable      Value            Expression
#air_red_MES  0.3              0.3
Description: maximum element size for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable      Value            Expression
#mu_flat_MAbE 30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable      Value            Expression
#mu_bent_MAbE 30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable      Value            Expression
#air_MAbE     30.0             30
Description: maximum angle between elements for air
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
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_left' UNIQUENAME='inner_brkt_bottom_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_door_BH FILE='bhdata//miniMSR_inner_door_anh.bh'
BHDATA label innermumetal_door_BH loaded with data from the file bhdata//miniMSR_inner_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_floor_BH FILE='bhdata//miniMSR_inner_floor_anh.bh'
BHDATA label innermumetal_floor_BH loaded with data from the file bhdata//miniMSR_inner_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Outer Layer
**** FILE INPUT ******
*** COMMENT ********** /++ / load structural step file
*** COMMENT ********** /++ LOAD OPTION=INSERT FILE='stpfiles//outer_layer.STEP'
*** COMMENT ********** /++
*** COMMENT ********** /++ /// Flat panels
*** COMMENT ********** /++ // front
*** COMMENT ********** /++ / rename body
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-001'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_front' UNIQUENAME='outer_front'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_front'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_front' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // back
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-026'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_back' UNIQUENAME='outer_back'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_back' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-014'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_left' UNIQUENAME='outer_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_left' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-018'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_right' UNIQUENAME='outer_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_right' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-025'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_top' UNIQUENAME='outer_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-005'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_bottom' UNIQUENAME='outer_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ /// brackets
*** COMMENT ********** /++ // front-top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-021'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_top' UNIQUENAME='outer_brkt_front_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-003'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_bottom' UNIQUENAME='outer_brkt_front_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-013'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_left' UNIQUENAME='outer_brkt_front_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-017'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_right' UNIQUENAME='outer_brkt_front_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-015'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_bottom_left' UNIQUENAME='outer_brkt_bottom_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-019'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_bottom_right' UNIQUENAME='outer_brkt_bottom_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // top-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-022'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_top_left' UNIQUENAME='outer_brkt_top_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // top-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-024'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_top_right' UNIQUENAME='outer_brkt_top_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-020'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_bottom' UNIQUENAME='outer_brkt_back_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-023'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_top' UNIQUENAME='outer_brkt_back_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-012'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_left' UNIQUENAME='outer_brkt_back_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-016'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_right' UNIQUENAME='outer_brkt_back_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // BH data
*** COMMENT ********** /++ / create labels
*** COMMENT ********** /++ BHDATA OPTION=NEW  LABEL='outermumetal_door_BH'
*** COMMENT ********** /++ BHDATA OPTION=NEW  LABEL='outermumetal_floor_BH'
*** COMMENT ********** /++
*** COMMENT ********** /++ / assign bh file to label
*** COMMENT ********** /++ BHDATA OPTION=LOAD LABEL=outermumetal_door_BH FILE='bhdata//miniMSR_outer_door_anh.bh'
*** COMMENT ********** /++ BHDATA OPTION=LOAD LABEL=outermumetal_floor_BH FILE='bhdata//miniMSR_outer_floor_anh.bh'
*** COMMENT ********** /++
*** COMMENT ********** /++ / match material label to bh label
*** COMMENT ********** /++ MATERIALS UNPICK
*** COMMENT ********** / MATERIALS GUIINIT
*** COMMENT ********** /++ MATERIALS PICK 'outer_mumetal_door'
*** COMMENT ********** /++ MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_door_BH'
*** COMMENT ********** / MATERIALS UNPICK
*** COMMENT ********** /++ MATERIALS UNPICK
*** COMMENT ********** / MATERIALS GUIINIT
*** COMMENT ********** /++ MATERIALS PICK 'outer_mumetal_floor'
*** COMMENT ********** /++ MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_floor_BH'
*** COMMENT ********** / MATERIALS UNPICK
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
**** FILE INPUT ****** BACKGROUND OPTION=LOAD
**** FILE INPUT ****** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=5 SCALEY=5 SCALEZ=5   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** //// Meshing
*** COMMENT ********** / Surface Mesh
**** FILE INPUT ****** MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
271 faces completed, generating 25211 surface elements

Triangular facet quality on 271 faces
Triangles: 25211, Max: 1.0, Min: 0.081491, Mean: 0.830982, Power2Mean: 0.367169

**** FILE INPUT ******
*** COMMENT ********** / Volume Mesh
**** FILE INPUT ****** FILL TOL=1.0E-05
Mesh Generator Type II Error Number : 5
 Error Message : Failed to generate tet mesh: Call to VKI function returned invalid data
vis_TetMeshGenerate : SYS_ERROR_OPERATION   An operation failed in current state : Input face intersection
Command file processing aborted
Check the log file or console for details.
Error in file: 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi' (Line 681)
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** COMMAND INPUT *** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
YES or NO
**** COMMAND INPUT *** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** YES
Information: modeller.comi was not found in the local or home directory
Command language initialization.
**** COMMAND INPUT ***
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\mesh_test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\mesh_test.comi
*** COMMENT ********** // Meshing Troubleshooting
*** COMMENT ********** // May 2024
*** COMMENT ********** // Start with only the bracket.
**** FILE INPUT ******
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_bracket.STEP'
Imported material name from label: Cut_Extrude3
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
End of command file.
*** CONTROL COMMAND ** $ END COMI
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=FACE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00020
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=0.01 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
41 faces completed, generating 7540 surface elements

Triangular facet quality on 41 faces
Triangles: 7540, Max: 0.999983, Min: 0.167946, Mean: 0.897384, Power2Mean: 0.562132

**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06

Tetrahedral element quality on 2 cells
Tetrahedra: 13734, Max: 1.597148, Min: 0.0546159, RMS: 0.858063, Power2Mean: 0.531603, Worst cell: 2

Total nodes:    3692
Tetrahedra:     13734
Total volume: 8.9801E-05
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\mesh_test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\mesh_test.comi
*** COMMENT ********** // Meshing Troubleshooting
*** COMMENT ********** // May 2024
*** COMMENT ********** // Start with only the bracket.
**** FILE INPUT ******
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_flat_front.STEP'
Imported material name from label: Boss_Extrude2
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
End of command file.
*** CONTROL COMMAND ** $ END COMI
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=FACE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00006
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=0.01 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
31 faces completed, generating 29320 surface elements

Triangular facet quality on 31 faces
Triangles: 29320, Max: 1.0, Min: 0.173763, Mean: 0.949609, Power2Mean: 0.686811

**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06

Tetrahedral element quality on 1 cells
Tetrahedra: 56187, Max: 1.561068, Min: 0.087821, RMS: 0.763053, Power2Mean: 0.462226, Worst cell: 1

Total nodes:    14527
Tetrahedra:     56187
Total volume: 5.84086E-04
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\mesh_test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\mesh_test.comi
*** COMMENT ********** // Meshing Troubleshooting
*** COMMENT ********** // May 2024
*** COMMENT ********** // Start with only the bracket.
**** FILE INPUT ******
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_bracket_flat.STEP'
Imported material name from label: inner_bracket_flat_NAUO1#inner_bracket_Cut_Extrude3
Imported material name from label: inner_bracket_flat_NAUO2#inner_flat_Boss_Extrude2
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
End of command file.
*** CONTROL COMMAND ** $ END COMI
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=FACE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-002' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00006
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00013
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00018
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00013
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00018
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=0.01 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
