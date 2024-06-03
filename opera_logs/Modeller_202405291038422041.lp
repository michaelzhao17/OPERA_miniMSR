Opera-3d Modeller                       started on 29/May/2024 at 10:38:42
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
*** COMMENT ********** /++ //// Create Model Body
*** COMMENT ********** /++ MODEL CREATE
*** COMMENT ********** /++
*** COMMENT ********** /++ //// Meshing
*** COMMENT ********** /++ / Surface Mesh
*** COMMENT ********** /++ MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
*** COMMENT ********** /++
*** COMMENT ********** /++ / Volume Mesh
*** COMMENT ********** /++ FILL TOL=1.0E-05
**** FILE INPUT ******
*** COMMENT ********** //// Create Database
*** COMMENT ********** /++ SOLVERS SOLVENOW=NO SAVEMODEL=YES,
*** COMMENT ********** / SOLVERS OPTION=TEST FILE='mini_MSR' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
*** COMMENT ********** / COMMENT CLEAR=YES TYPE=DBTITLE
*** COMMENT ********** / SOLVERS OPTION=OVERWRITE
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
**** COMMAND INPUT *** FILTER COMMAND=CURSORHIDE
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER COMMAND=PICK
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
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
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.05, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.05             0.05
Description: maximum element size for bent mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.05, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.05             0.05
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
*** COMMENT ********** /++ //// Create Model Body
*** COMMENT ********** /++ MODEL CREATE
*** COMMENT ********** /++
*** COMMENT ********** /++ //// Meshing
*** COMMENT ********** /++ / Surface Mesh
*** COMMENT ********** /++ MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
*** COMMENT ********** /++
*** COMMENT ********** /++ / Volume Mesh
*** COMMENT ********** /++ FILL TOL=1.0E-05
**** FILE INPUT ******
*** COMMENT ********** //// Create Database
*** COMMENT ********** /++ SOLVERS SOLVENOW=NO SAVEMODEL=YES,
*** COMMENT ********** / SOLVERS OPTION=TEST FILE='mini_MSR' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
*** COMMENT ********** / COMMENT CLEAR=YES TYPE=DBTITLE
*** COMMENT ********** / SOLVERS OPTION=OVERWRITE
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
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL DELETE
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=LOAD
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=3 SCALEY=3 SCALEZ=3   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $OS OPERAPOST BACK ''
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
271 faces completed, generating 20847 surface elements

Triangular facet quality on 271 faces
Triangles: 20847, Max: 1.0, Min: 0.117811, Mean: 0.798593, Power2Mean: 0.337991

**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06
Mesh Generator Type II Error Number : 5
 Error Message : Failed to generate tet mesh: Call to VKI function returned invalid data
vis_TetMeshGenerate : SYS_ERROR_OPERATION   An operation failed in current state : Input face intersection
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL DELETE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=MOSAIC
Mesh generator Type I
Initial triangulation contained 3993 nodes.
271 faces completed, generating 9826 surface elements

Triangular facet quality on 271 faces
Triangles: 9826, Max: 0.999959, Min: 0.0322381, Mean: 0.692877, Power2Mean: 0.154753

Face 1 is marked 'Surface mesh warning'
Face 2 is marked 'Surface mesh warning'
Face 8 is marked 'Surface mesh warning'
Face 12 is marked 'Surface mesh warning'
Face 14 is marked 'Surface mesh warning'
Face 18 is marked 'Surface mesh warning'
Face 20 is marked 'Surface mesh warning'
Face 22 is marked 'Surface mesh warning'
Face 76 is marked 'Surface mesh warning'
Face 93 is marked 'Surface mesh warning'
Face 96 is marked 'Surface mesh warning'
Face 100 is marked 'Surface mesh warning'
Face 193 is marked 'Surface mesh warning'
Face 197 is marked 'Surface mesh warning'
Face 203 is marked 'Surface mesh warning'
Face 217 is marked 'Surface mesh warning'
Checking triangular mesh on 271 faces issued warning for 16 faces
Checking triangular mesh on faces issued warning
You can go forward, while you are advised to check flagged faces
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06

Tetrahedral element quality on 15 cells
Tetrahedra: 81787, Max: 1.869887, Min: 0.0013464, RMS: 1.022104, Power2Mean: 0.238357, Worst cell: 10

Total nodes:    14682
Tetrahedra:     81787
Total volume: 1.879504
Possible surface mesh issue: please check the surface mesh on Cell 10, particularly faces touching Face 100.
Especially for thin triangles caused by small details compared to the element size.
Possible surface mesh issue: please check the surface mesh on Cell 10, particularly faces touching Face 126.
Possible surface mesh issue: please check the surface mesh on Cell 10, particularly faces touching Face 150.
Possible surface mesh issue: please check the surface mesh on Cell 10, particularly faces touching Face 168.
Possible surface mesh issue: please check the surface mesh on Cell 10, particularly faces touching Face 178.
Possible surface mesh issue: please check the surface mesh on Cell 10, particularly faces touching Face 197.
Possible surface mesh issue: please check the surface mesh on Cell 10, particularly faces touching Face 241.
Possible issue when volume meshing Cell 10: please check the surface mesh on this cell.
**** COMMAND INPUT ***
**** COMMAND INPUT *** SOLVERS SOLVENOW=NO SAVEMODEL=YES,
**** COMMAND INPUT *** SOLVERS OPTION=TEST FILE='layer_mesh' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
**** COMMAND INPUT *** COMMENT CLEAR=YES TYPE=DBTITLE
**** COMMAND INPUT *** SOLVERS OPTION=OVERWRITE
Opening database layer_mesh.op3, simulation number 0 on 29/May/2024 at 10:43:36
   Magnetostatic Analysis (TOSCA)

   File: layer_mesh.op3 simulation:  1
   Created on: 29/May/2024 10:43:35
   In Directory:
   C:\Users\mzhao\OPERA_miniMSR\
   By Machine:
   Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009. version windows (10.0.19045)
   Log Files: Modeller_202405291038422041.log/lp
   Simulation created using: Opera-3d/Modeller Version 21.3.3
   User did not enter a title

   SI units

   No conductors
   Adaptive RHS integrals

   Automatic potential cuts are enabled.

   Boundary conditions:
     NORMMAGN
     TANGMAGN
        Note: Assumed as the default boundary condition to one or more external faces.

   Linear solution

   No periodicity conditions have been defined

   1 solution scaling factor defined for all drives: 1.0

   Materials defined:
   INNER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_inner_door_anh.bh (15402.39)
   INNER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_inner_floor_anh.bh (25009.79)
   AIR

   35322 nodes in the model
   106099 edges in the model
   Both linear and quadratic elements exist in the model
   81787 tetrahedral elements (72354 linear and 9433 quadratic)
   81787 elements in total

   Symmetry:
   Positive reflection in YZ plane
   Negative reflection in ZX plane
**** COMMAND INPUT ***
**** COMMAND INPUT *** MULTIPHYSICS OPTION=RESET
**** COMMAND INPUT ***
**** COMMAND INPUT *** ANALYSISDATA OPTION=SET PROGRAM=TOSCAMAGN HX=0 HY=50 HZ=0 LINEAR=NO NITERATIONS=21 NLITERTYPE=NEWTON POTENTIALCUT=YES RHS=ADAPTIVE SCALEDRIVE=ALL TOLERANCE=0.001 USEDEFORMEDMESH=NO USEDIRECTSOLVER=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** SOLVERS SOLVENOW=NO SAVEMODEL=YES,
**** COMMAND INPUT *** SOLVERS OPTION=TEST FILE='layer_mesh.op3' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
Files already exist:
     layer_mesh.op3
     layer_mesh.opc
Do you want to overwrite?
YES or NO
*** LOOP COMMAND ***** yes
**** COMMAND INPUT *** COMMENT CLEAR=YES TYPE=DBTITLE
**** COMMAND INPUT *** SOLVERS OPTION=OVERWRITE
Opening database layer_mesh.op3, simulation number 0 on 29/May/2024 at 10:44:10
   Magnetostatic Analysis (TOSCA)

   File: layer_mesh.op3 simulation:  1
   Created on: 29/May/2024 10:44:09
   In Directory:
   C:\Users\mzhao\OPERA_miniMSR\
   By Machine:
   Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009. version windows (10.0.19045)
   Log Files: Modeller_202405291038422041.log/lp
   Simulation created using: Opera-3d/Modeller Version 21.3.3
   User did not enter a title

   SI units

   No conductors
   External field vector (0.0,50.0,0.0 A/m)
   Adaptive RHS integrals

   Automatic potential cuts are enabled.

   Boundary conditions:
     NORMMAGN
     TANGMAGN
        Note: Assumed as the default boundary condition to one or more external faces.

   Nonlinear iteration data
   Newton-Raphson Iterations
   Maximum Iterations: 21
   Tolerance: 1.0E-03

   No periodicity conditions have been defined

   1 solution scaling factor defined for all drives: 1.0

   Materials defined:
   INNER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_inner_door_anh.bh (15402.39)
   INNER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_inner_floor_anh.bh (25009.79)
   AIR

   35322 nodes in the model
   106099 edges in the model
   Both linear and quadratic elements exist in the model
   81787 tetrahedral elements (72354 linear and 9433 quadratic)
   81787 elements in total

   Symmetry:
   Positive reflection in YZ plane
   Negative reflection in ZX plane
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=LOAD
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL DELETE
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
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.05, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.05             0.05
Description: maximum element size for bent mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.05, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.05             0.05
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
*** COMMENT ********** /++ //// Create Model Body
*** COMMENT ********** /++ MODEL CREATE
*** COMMENT ********** /++
*** COMMENT ********** /++ //// Meshing
*** COMMENT ********** /++ / Surface Mesh
*** COMMENT ********** /++ MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
*** COMMENT ********** /++
*** COMMENT ********** /++ / Volume Mesh
*** COMMENT ********** /++ FILL TOL=1.0E-05
**** FILE INPUT ******
*** COMMENT ********** //// Create Database
*** COMMENT ********** /++ SOLVERS SOLVENOW=NO SAVEMODEL=YES,
*** COMMENT ********** / SOLVERS OPTION=TEST FILE='mini_MSR' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
*** COMMENT ********** / COMMENT CLEAR=YES TYPE=DBTITLE
*** COMMENT ********** / SOLVERS OPTION=OVERWRITE
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
**** COMMAND INPUT *** MODEL CREATE
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
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.05, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.05             0.05
Description: maximum element size for bent mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.05, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.05             0.05
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
**** FILE INPUT ****** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=2 SCALEY=2 SCALEZ=2   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** /++ //// Meshing
*** COMMENT ********** /++ / Surface Mesh
*** COMMENT ********** /++ MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
*** COMMENT ********** /++
*** COMMENT ********** /++ / Volume Mesh
*** COMMENT ********** /++ FILL TOL=1.0E-05
**** FILE INPUT ******
*** COMMENT ********** //// Create Database
*** COMMENT ********** /++ SOLVERS SOLVENOW=NO SAVEMODEL=YES,
*** COMMENT ********** / SOLVERS OPTION=TEST FILE='mini_MSR' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
*** COMMENT ********** / COMMENT CLEAR=YES TYPE=DBTITLE
*** COMMENT ********** / SOLVERS OPTION=OVERWRITE
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
**** COMMAND INPUT *** MULTIPHYSICS OPTION=RESET
**** COMMAND INPUT ***
**** COMMAND INPUT *** ANALYSISDATA OPTION=SET PROGRAM=TOSCAMAGN HX=0 HY=50 HZ=0 LINEAR=NO NITERATIONS=21 NLITERTYPE=NEWTON POTENTIALCUT=YES RHS=ADAPTIVE SCALEDRIVE=ALL TOLERANCE=0.001 USEDEFORMEDMESH=NO USEDIRECTSOLVER=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=MOSAIC
Mesh generator Type I
Initial triangulation contained 8258 nodes.
786 faces completed, generating 23179 surface elements

Triangular facet quality on 786 faces
Triangles: 23179, Max: 0.99998, Min: 0.0148552, Mean: 0.546182, Power2Mean: 0.078715

Face 1 is marked 'Surface mesh warning'
Face 2 is marked 'Surface mesh warning'
Face 46 is marked 'Surface mesh warning'
Face 59 is marked 'Surface mesh warning'
Face 73 is marked 'Surface mesh warning'
Face 77 is marked 'Surface mesh warning'
Face 78 is marked 'Surface mesh warning'
Face 100 is marked 'Surface mesh warning'
Face 103 is marked 'Surface mesh warning'
Face 109 is marked 'Surface mesh warning'
Face 126 is marked 'Surface mesh warning'
Face 128 is marked 'Surface mesh warning'
Face 130 is marked 'Surface mesh warning'
Face 134 is marked 'Surface mesh warning'
Face 136 is marked 'Surface mesh warning'
Face 140 is marked 'Surface mesh warning'
Face 143 is marked 'Surface mesh warning'
Face 147 is marked 'Surface mesh warning'
Face 150 is marked 'Surface mesh warning'
Face 151 is marked 'Surface mesh warning'
Face 155 is marked 'Surface mesh warning'
Face 156 is marked 'Surface mesh warning'
Face 157 is marked 'Surface mesh warning'
Face 158 is marked 'Surface mesh warning'
Face 159 is marked 'Surface mesh warning'
Face 163 is marked 'Surface mesh warning'
Face 164 is marked 'Surface mesh warning'
Face 165 is marked 'Surface mesh warning'
Face 167 is marked 'Surface mesh warning'
Face 168 is marked 'Surface mesh warning'
Face 170 is marked 'Surface mesh warning'
Face 183 is marked 'Surface mesh warning'
Face 192 is marked 'Surface mesh warning'
Face 199 is marked 'Surface mesh warning'
Face 208 is marked 'Surface mesh warning'
Face 212 is marked 'Surface mesh warning'
Face 217 is marked 'Surface mesh warning'
Face 227 is marked 'Surface mesh warning'
Face 239 is marked 'Surface mesh warning'
Face 247 is marked 'Surface mesh warning'
Face 253 is marked 'Surface mesh warning'
Face 255 is marked 'Surface mesh warning'
Face 258 is marked 'Surface mesh warning'
Face 259 is marked 'Surface mesh warning'
Face 260 is marked 'Surface mesh warning'
Face 262 is marked 'Surface mesh warning'
Face 263 is marked 'Surface mesh warning'
Face 264 is marked 'Surface mesh warning'
Face 276 is marked 'Surface mesh warning'
Face 284 is marked 'Surface mesh warning'
Face 289 is marked 'Surface mesh warning'
Face 296 is marked 'Surface mesh warning'
Face 300 is marked 'Surface mesh warning'
Face 305 is marked 'Surface mesh warning'
Face 316 is marked 'Surface mesh warning'
Face 327 is marked 'Surface mesh warning'
Face 332 is marked 'Surface mesh warning'
Face 336 is marked 'Surface mesh warning'
Face 337 is marked 'Surface mesh warning'
Face 340 is marked 'Surface mesh warning'
Face 341 is marked 'Surface mesh warning'
Face 342 is marked 'Surface mesh warning'
Face 344 is marked 'Surface mesh warning'
Face 345 is marked 'Surface mesh warning'
Face 346 is marked 'Surface mesh warning'
Face 358 is marked 'Surface mesh warning'
Face 366 is marked 'Surface mesh warning'
Face 371 is marked 'Surface mesh warning'
Face 378 is marked 'Surface mesh warning'
Face 382 is marked 'Surface mesh warning'
Face 385 is marked 'Surface mesh warning'
Face 397 is marked 'Surface mesh warning'
Face 409 is marked 'Surface mesh warning'
Face 413 is marked 'Surface mesh warning'
Face 418 is marked 'Surface mesh warning'
Face 419 is marked 'Surface mesh warning'
Face 422 is marked 'Surface mesh warning'
Face 423 is marked 'Surface mesh warning'
Face 424 is marked 'Surface mesh warning'
Face 426 is marked 'Surface mesh warning'
Face 427 is marked 'Surface mesh warning'
Face 428 is marked 'Surface mesh warning'
Face 441 is marked 'Surface mesh warning'
Face 451 is marked 'Surface mesh warning'
Face 453 is marked 'Surface mesh warning'
Face 462 is marked 'Surface mesh warning'
Face 464 is marked 'Surface mesh warning'
Face 467 is marked 'Surface mesh warning'
Face 477 is marked 'Surface mesh warning'
Face 489 is marked 'Surface mesh warning'
Face 578 is marked 'Surface mesh warning'
Face 583 is marked 'Surface mesh warning'
Face 590 is marked 'Surface mesh warning'
Face 594 is marked 'Surface mesh warning'
Face 598 is marked 'Surface mesh warning'
Face 608 is marked 'Surface mesh warning'
Face 616 is marked 'Surface mesh warning'
Face 632 is marked 'Surface mesh warning'
Face 634 is marked 'Surface mesh warning'
Face 635 is marked 'Surface mesh warning'
Face 639 is marked 'Surface mesh warning'
Face 644 is marked 'Surface mesh warning'
Face 647 is marked 'Surface mesh warning'
Face 649 is marked 'Surface mesh warning'
Face 651 is marked 'Surface mesh warning'
Face 654 is marked 'Surface mesh warning'
Face 658 is marked 'Surface mesh warning'
Face 708 is marked 'Surface mesh warning'
Face 712 is marked 'Surface mesh warning'
Face 718 is marked 'Surface mesh warning'
Face 732 is marked 'Surface mesh warning'
Checking triangular mesh on 786 faces issued warning for 111 faces
Checking triangular mesh on faces issued warning
You can go forward, while you are advised to check flagged faces
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06
Possible issue when volume meshing Cell 18: please check the surface mesh on this cell.
Especially for thin triangles caused by small details compared to the element size.
Mesh generator error code: 401 in cell 19 of 32.
RestorV15: 3 missing facets in volume mesh
Use the CHECK command to identify any geometric errors
Cell 19 is marked 'Volume mesh error'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** !CHECK
Allow commands to be previewed before accepting any changes
!CHECK is an ambiguous or invalid choice for parameter OPTION. Options are:
Start, End, Pick, Accept, Unpick
**** COMMAND INPUT ***
**** COMMAND INPUT *** CHECK !
Check for topological problems with entities
Parameter Value                   Meaning
LEVEL     HIGH                    Level of checking: FULL, HIGH, MEDIUM or LOW
REPAIR    NO                      Attempt to repair the edges and vertices of the body
FILENAME                          Output the results of checking to a data file
**** COMMAND INPUT ***
**** COMMAND INPUT *** CHECK LEVEL=HIGH REPAIR=YES
Bodies must be picked for checking
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=BODY
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='ModelBody'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** CHECK LEVEL=HIGH
Repair cannot be used on the model body
There  are 0 errors in the body checked
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PICK OPTION=RESET
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='ModelBody'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL DELETE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** CHECK LEVEL=HIGH REPAIR=YES
There  are 0 errors in the body checked
There are 36 tolerant entities in the body
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** CHECK LEVEL=HIGH REPAIR=YES
There  are 0 errors in the body checked
There are 36 tolerant entities in the body
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** CHECK LEVEL=HIGH REPAIR=YES
There  are 0 errors in the body checked
There are 52 tolerant entities in the body
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=BODY
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER COMMAND=CURSORHIDE
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE PROPERTY=UniqueName LABEL='outer_right'
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE PROPERTY=UniqueName LABEL='Unnamed:AAB-010'
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE PROPERTY=UniqueName LABEL='inner_right'
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER COMMAND=PICK
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** CHECK LEVEL=HIGH REPAIR=YES
There  are 0 errors in the body checked
There are 52 tolerant entities in the body
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** CHECK LEVEL=HIGH REPAIR=YES
There  are 0 errors in the body checked
There are 52 tolerant entities in the body
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** CHECK LEVEL=HIGH REPAIR=YES
There  are 0 errors in the body checked
There are 42 tolerant entities in the body
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
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.05, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.05             0.05
Description: maximum element size for bent mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.05, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.05             0.05
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
*** COMMENT ********** /++ / load structural step file
*** COMMENT ********** /++ LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
*** COMMENT ********** /++
*** COMMENT ********** /++ /// Flat panels
*** COMMENT ********** /++ // front
*** COMMENT ********** /++ / rename body
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // back
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ /// brackets
*** COMMENT ********** /++ // front-top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_bottom_left' UNIQUENAME='inner_brkt_bottom_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // top-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // top-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
*** COMMENT ********** /++
*** COMMENT ********** /++ // BH data
*** COMMENT ********** /++ / create labels
*** COMMENT ********** /++ BHDATA OPTION=NEW  LABEL='innermumetal_door_BH'
*** COMMENT ********** /++ BHDATA OPTION=NEW  LABEL='innermumetal_floor_BH'
*** COMMENT ********** /++
*** COMMENT ********** /++ / assign bh file to label
*** COMMENT ********** /++ BHDATA OPTION=LOAD LABEL=innermumetal_door_BH FILE='bhdata//miniMSR_inner_door_anh.bh'
*** COMMENT ********** /++ BHDATA OPTION=LOAD LABEL=innermumetal_floor_BH FILE='bhdata//miniMSR_inner_floor_anh.bh'
*** COMMENT ********** /++
*** COMMENT ********** /++ / match material label to bh label
*** COMMENT ********** /++ MATERIALS UNPICK
*** COMMENT ********** / MATERIALS GUIINIT
*** COMMENT ********** /++ MATERIALS PICK 'inner_mumetal_door'
*** COMMENT ********** /++ MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_door_BH'
*** COMMENT ********** / MATERIALS UNPICK
*** COMMENT ********** /++ MATERIALS UNPICK
*** COMMENT ********** / MATERIALS GUIINIT
*** COMMENT ********** /++ MATERIALS PICK 'inner_mumetal_floor'
*** COMMENT ********** /++ MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_floor_BH'
*** COMMENT ********** / MATERIALS UNPICK
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
Options available for parameter LABEL are:
Unnamed:AAA-001, Unnamed:AAA-002, Unnamed:AAA-003, Unnamed:AAA-004, Unnamed:AAA-005, Unnamed:AAA-006, Unnamed:AAA-007, Unnamed:AAA-008, Unnamed:AAA-009, Unnamed:AAA-010, Unnamed:AAA-011, Unnamed:AAA-012, Unnamed:AAA-013, Unnamed:AAA-014, Unnamed:AAA-015, Unnamed:AAA-016, Unnamed:AAA-017, Unnamed:AAA-018, Unnamed:AAA-019, Unnamed:AAA-020, Unnamed:AAA-021, Unnamed:AAA-022, Unnamed:AAA-023, Unnamed:AAA-024, Unnamed:AAA-025, Unnamed:AAA-026
Invalid label specified
Command file processing aborted
Check the log file or console for details.
Error in file: 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi' (Line 365)
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-026'
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
