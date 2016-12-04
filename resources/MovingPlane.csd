<?xml version="1.0"?>
<CsoundSynthesizer>
<CsOptions>
-o dac0
-+rtaudio=PortAudio
-B 2048
-b 32
</CsOptions>
<CsInstruments>

sr 		= 	44100	;SAMPLE RATE
ksmps 	= 	32	;NUMBER OF AUDIO SAMPLES IN EACH CONTROL CYCLE
nchnls 	= 	2	;NUMBER OF CHANNELS (2=STEREO)
0dbfs	=	1	;MAXIMUM AMPLITUDE REGARDLESS OF BIT DEPTH

#define HRTF_DB_LEFT		# "..\..\hrtf\hrtf-44100-left.dat" #
#define HRTF_DB_RIGHT		# "..\..\hrtf\hrtf-44100-right.dat" #
#define HRTF_DB				# "..\..\hrtf\hrtf-44100-left.dat", "..\..\hrtf\hrtf-44100-right.dat" #

#define TEST_INSTRUMENT(ID)
#
	instr $ID
		aout oscili 1, 440, 100
		outs aout, aout
	endin
#

#define MULTI_SPEAKER_OUT(AZIMUTH' ELEVATION' MOD360)
#
	kAmpTop = sqrt(($ELEVATION + 30) / 60)
	kAmpBot = sqrt(1 - kAmpTop * kAmpTop)
	
	kAmpFront = 1
	kAmpBack = 0

	if($MOD360 >= 1) then
		kAmpFront = (180 - (abs($AZIMUTH))) / 180
		kAmpBack = 1 - kAmpFront
	endif
		
	outch 1, aleft	* kAmpTop * kAmpFront
	outch 2, aright	* kAmpTop * kAmpFront

	outch 3, aleft	* kAmpTop * kAmpBack
	outch 4, aright	* kAmpTop * kAmpBack
	outch 5, aleft	* kAmpBot * kAmpBack
	outch 6, aright	* kAmpBot * kAmpBack

	outch 7, aleft	* kAmpBot * kAmpFront
	outch 8, aright	* kAmpBot * kAmpFront
#

#define MULTI_SPEAKER_IN(ID' AZIMUTH' ELEVATION' VOLUME 'MOD360)
#
	$VOLUME				init	1 
	$VOLUME				chnget 	"vol$ID"

	$AZIMUTH 			init 	0
	$AZIMUTH 			chnget 	"azimuth$ID"

	$ELEVATION 			init 	0
	$ELEVATION			chnget 	"elev$ID"

	$MOD360				init 	0	
	$MOD360				chnget 	"d360_$ID"
#

#define BARMODEL_INSTRUMENT(ID)				
# 
instr $ID

	$MULTI_SPEAKER_IN($ID' kAzimuth' kElevation' kVol' kDeg360)

	idur				init	p3
	iamp				init	p4
	kboundarycondl 		init 	p5
	kboundarycondr 		init 	p6
	istiff				init 	p7 
	ihighfreqloss		init	p8
	kscanspeed			init	p9
	idecay30db			init	p10
	ivel				init	p11
	iwid 				init	p12
	idecaystartperc		init	p13 ; if 0 then no envelope
	idanger				init 	p14
	ielev				init  	p15
	ipos				init	0.5
	
	kExtDecay			init    1

	kMute				init 	1
	kMute				chnget	"mute$ID"

	kTremAmp			init 	0
	kTremAmp			chnget 	"tremAmp$ID"

	kTremFreq			init	0
	kTremFreq			chnget 	"tremFreq$ID"
	
	kbend				line	p9, idur, p9 + (ielev / 100.0)
	
	if (kMute >= 1) kgoto decay
		kgoto play

	decay:
		if(kExtDecay >= 0.02) then 
			kExtDecay = kExtDecay - 0.02
		else
			kExtDecay = 0
			
		endif
	
	play:
	kAmp				expseg	iamp,		idur*idecaystartperc, 		iamp, 		idur*(1-idecaystartperc)-0.01, 	0.001

	;   boundary conditions: 1 - clamped, 2- pivoting, 3 - free. When free is used on one end, the other must be clamped, otherwise the model will "blow up".
	;					LEFT      		| RIGHT     	| STIFFNESS | HIGH-      	|  SCANNING   | 30dB   		| STRIKE   | STRIKE   | WIDTH
	;					BOUNDARY  		| BOUNDARY  	|           | -FREQUENCY 	|  SPEED      | DECAY  		| POSITION | VELOCITY | OF
	;					CONDITION 		| CONDITION 	|           | LOSS       	|             | TIME   		|          |          | STRIKE
	ares	barmodel	kboundarycondl, kboundarycondr,    istiff,    ihighfreqloss,  kscanspeed,   idecay30db,     ipos,      ivel,      iwid

	kSrc = kAmp

	if (idanger >= 1) then
		ktrem	oscil   kTremAmp, kTremFreq, 21
		kSrc = kSrc + ktrem
	endif

	asrc = kSrc * ares * kVol * kExtDecay
	aleft, aright hrtfmove asrc, kAzimuth, kElevation, $HRTF_DB
	
;	printks "barmodel: %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t \n", 0.5, \\
;	kboundarycondl, kboundarycondr, istiff, ihighfreqloss, kbend, idecay30db, ipos, ivel, iwid

	$MULTI_SPEAKER_OUT(kAzimuth' kElevation' kDeg360)

endin
#

#define BARMODEL_INSTRUMENT2(ID)
#
instr $ID

	$MULTI_SPEAKER_IN($ID' kAzimuth' kElevation' kVol' kDeg360)

	iVol				init	p4
	iMode				init	p5

	asig loscil 1, 1, p5, 1, 0

	aleft, aright hrtfmove iVol * kVol * asig, kAzimuth, kElevation, $HRTF_DB

	$MULTI_SPEAKER_OUT(kAzimuth' kElevation' kDeg360)

endin
#

#define BARMODEL_INSTRUMENT3(ID' MULTI_SPEAKER)
#
instr $ID

	$MULTI_SPEAKER_IN($ID' kAzimuth' kElevation' kVol' kDeg360)
	
	iVol				init	p4
	iMode				init	p5		

	if (4 > iMode) then
		asig loscil 1, 1, iMode + 1, 1, 0
	else
		asig loscil 1, 1, 4, 1, 0
	endif
	
	aleft, aright hrtfmove iVol * kVol * asig, kAzimuth, kElevation, $HRTF_DB

	if ($MULTI_SPEAKER > 0) then
		$MULTI_SPEAKER_OUT(kAzimuth' kElevation' kDeg360)
	else
		outs aleft, aright
	endif
		
endin
#

#define BARMODEL_INSTRUMENT4(ID)
#
instr $ID
	
	$MULTI_SPEAKER_IN($ID' kAzimuth' kElevation' kVol' kDeg360)

	iVol				init	p4

	kTurnOff			chnget 	"turnoff$ID"

	if kTurnOff == 0 kgoto cont
	
	turnoff
	
	cont:	
		asig loscil 1, 1, 10, 1, 1
		ihold

	aleft, aright hrtfmove iVol * kVol * asig, kAzimuth, kElevation, $HRTF_DB
	
	$MULTI_SPEAKER_OUT(kAzimuth' kElevation' kDeg360)

endin
#

#define BARMODEL_INSTRUMENT5(ID)
#
instr $ID
	;open space marker
	
	$MULTI_SPEAKER_IN($ID' kAzimuth' kElevation' kVol' kDeg360)
	
	idur				init	p3
	iamp				init	p4
	iVol				init	p5
	iAzimuthStop		init 	p6
	iFadeInPerc			init 	p7
	iFadeOutPerc		init 	p8
	iFreq				init 	p9
	
	kAmp				linen	iamp,	idur*iFadeInPerc, 		idur, 		idur*(iFadeOutPerc)
	kAzMove				line	0.01, 	idur, 	iAzimuthStop
	
	asig oscili 1, iFreq, 51
	
	asrc = iVol * asig * kAmp 
	aleft, aright hrtfmove asrc, kAzMove, kElevation, $HRTF_DB
	
	$MULTI_SPEAKER_OUT(kAzMove' kElevation' kDeg360)

endin
#

$BARMODEL_INSTRUMENT(1)
$BARMODEL_INSTRUMENT(2)
$BARMODEL_INSTRUMENT(3)
$BARMODEL_INSTRUMENT(4)
$BARMODEL_INSTRUMENT(5)
$BARMODEL_INSTRUMENT(6)
$BARMODEL_INSTRUMENT(7)
$BARMODEL_INSTRUMENT(8)
$BARMODEL_INSTRUMENT(9)
$BARMODEL_INSTRUMENT(10)
$BARMODEL_INSTRUMENT(11)
$BARMODEL_INSTRUMENT(12)
$BARMODEL_INSTRUMENT(13)
$BARMODEL_INSTRUMENT(14)
$BARMODEL_INSTRUMENT(15)
$BARMODEL_INSTRUMENT(16)
$BARMODEL_INSTRUMENT(17)
$BARMODEL_INSTRUMENT(18)
$BARMODEL_INSTRUMENT(19)
$BARMODEL_INSTRUMENT(20)

$BARMODEL_INSTRUMENT2(21)
$BARMODEL_INSTRUMENT2(22)
$BARMODEL_INSTRUMENT2(23)


$BARMODEL_INSTRUMENT3(31' 1)
$BARMODEL_INSTRUMENT3(32' 1)
$BARMODEL_INSTRUMENT3(33' 0) ; this one might not requre full 8 channel output

$BARMODEL_INSTRUMENT4(41)

$BARMODEL_INSTRUMENT5(51)

</CsInstruments>
<CsScore>
f 1 0 0 1 "hole.wav" 0 0 0
f 2 0 0 1 "upstairs.wav" 0 0 0
f 3 0 0 1 "downstairs.wav" 0 0 0
f 4 0 0 1 "door.wav" 0 0 0
f 10 0 0 1 "ring.wav" 0 0 0 
f 30 0 0 1 "marker-percuss01.wav" 0 0 0
f 31 0 0 1 "marker-percuss02.wav" 0 0 0
f 32 0 0 1 "marker-percuss03.wav" 0 0 0
i1 0 10000
f 21 0 4096 10 1
f 51 0 1024 10 1

; signal table for TEST_INSTRUMENT
f 100 0 128 10 1

;				start	duration 	amplitude	   left    right 	STIFFNESS    highfreq-loss    scanning speed    30db-decay       strike velocity 	width    decay start (%)
;				TIME 							  boundary boundary


;i "barmodel" 	0  			0.5			    2   		1   		1 		64 	   		.2     			.01    				50 				1600 		0.5 			0.16
; 262 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		72 	   		.     			.    				. 				. 			. 			.
; 294 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		80 	   		.     			.    				. 				. 			. 			.
; 330 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		84 	   		.    			.    				. 				. 			. 			.
; 349 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		94 	   		.     			.    				. 				. 			. 			.
; 392 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		105 	   	.     			.    				. 				. 			. 			.
; 440 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		117 	   	.     			.    				. 				. 			. 			.
; 494 Hz			
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		123 	   	.     			.    				. 				. 			. 			.			
; 523 Hz
;i "barmodel" 	^+1 		0.5			    2   		1   		1 		64 	   		.2     			.5    				50 				1600 		0.5 			0.16
; 262 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		72 	   		.     			.    				. 				. 			. 			.
; 294 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		80 	   		.     			.    				. 				. 			. 			.
; 330 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		84 	   		.    			.    				. 				. 			. 			.
; 349 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		94 	   		.     			.    				. 				. 			. 			.
; 392 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		105 	   	.     			.    				. 				. 			. 			.
; 440 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		117 	   	.     			.    				. 				. 			. 			.
; 494 Hz			
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		123 	   	.     			.    				. 				. 			. 			.			
; 523 Hz
;i "barmodel" 	^+1 		0.5			    2   		1   		1 		64 	   		.2     			1    				50 				1600 		0.5 			0.16
; 262 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		72 	   		.     			.    				. 				. 			. 			.
; 294 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		80 	   		.     			.    				. 				. 			. 			.
; 330 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		84 	   		.    			.    				. 				. 			. 			.
; 349 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		94 	   		.     			.    				. 				. 			. 			.
; 392 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		105 	   	.     			.    				. 				. 			. 			.
; 440 Hz
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		117 	   	.     			.    				. 				. 			. 			.
; 494 Hz			
;i "barmodel" 	^+0.5 		.			    .   		.   		. 		123 	   	.     			.    				. 				. 			. 			.			
; 523 Hz

</CsScore>
</CsoundSynthesizer>
