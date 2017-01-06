<CsoundSynthesizer>
<CsOptions>
-o dac:hw:1,0
-b 32
-m0
</CsOptions>
<CsInstruments>

sr 		= 	44100	;SAMPLE RATE
ksmps 	= 	32	;NUMBER OF AUDIO SAMPLES IN EACH CONTROL CYCLE
nchnls 	= 	2	;NUMBER OF CHANNELS (2=STEREO)
0dbfs	=	1	;MAXIMUM AMPLITUDE REGARDLESS OF BIT DEPTH

instr 1

	kVol				init	1 
	kVol				chnget 	"vol1"

	kAzimuth 			init 	0
	kAzimuth 			chnget 	"azimuth1"

	kElevation 			init 	0
	kElevation			chnget 	"elev1"

	idur				init	p3
	iamp				init	p4
	kboundarycondl 			init 	p5
	kboundarycondr 			init 	p6
	istiff				init 	p7 
	ihighfreqloss			init	p8
	kscanspeed			init	p9
	idecay30db			init	p10
	ivel				init	p11
	iwid 				init	p12
	idecaystartperc			init	p13 ; if 0 then no envelope
	idanger				init 	p14
	ielev				init  	p15
	ipos				init	0.5
	
	kExtDecay			init    1

	kMute				init 	1
	kMute				chnget	"mute1"

	kTremAmp			init 	0
	kTremAmp			chnget 	"tremAmp1"

	kTremFreq			init	0
	kTremFreq			chnget 	"tremFreq1"
	
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
	kAmp	expseg	iamp,	idur*idecaystartperc, 	iamp,	idur*(1-idecaystartperc)-0.01, 	0.001

	;   boundary conditions: 1 - clamped, 2- pivoting, 3 - free. When free is used on one end, the other must be clamped, otherwise the model will "blow up".
	;					LEFT      		| RIGHT     	| STIFFNESS | HIGH-      	|  SCANNING   | 30dB   		| STRIKE   | STRIKE   | WIDTH
	;					BOUNDARY  		| BOUNDARY  	|           | -FREQUENCY 	|  SPEED      | DECAY  		| POSITION | VELOCITY | OF
	;					CONDITION 		| CONDITION 	|           | LOSS       	|             | TIME   		|          |          | STRIKE
	ares	barmodel	kboundarycondl, kboundarycondr,    istiff,    ihighfreqloss,  kscanspeed,   idecay30db,     ipos,      ivel,      iwid

	kSrc = kAmp

	asrc = kSrc * ares * kVol * kExtDecay
	aleft, aright hrtfmove asrc, kAzimuth, kElevation, "hrtf\hrtf-44100-left.dat", "hrtf\hrtf-44100-right.dat"
	
;	printks "barmodel: %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t \n", 0.5, \\
;	kboundarycondl, kboundarycondr, istiff, ihighfreqloss, kbend, idecay30db, ipos, ivel, iwid

	out aleft, aright

endin

instr 2

	kVol				init	1 
	kVol				chnget 	"vol2"

	kAzimuth 			init 	0
	kAzimuth 			chnget 	"azimuth2"

	kElevation 			init 	0
	kElevation			chnget 	"elev2"

	idur				init	p3
	iamp				init	p4
	kboundarycondl 			init 	p5
	kboundarycondr 			init 	p6
	istiff				init 	p7 
	ihighfreqloss			init	p8
	kscanspeed			init	p9
	idecay30db			init	p10
	ivel				init	p11
	iwid 				init	p12
	idecaystartperc			init	p13 ; if 0 then no envelope
	idanger				init 	p14
	ielev				init  	p15
	ipos				init	0.5
	
	kExtDecay			init    1

	kMute				init 	1
	kMute				chnget	"mute2"

	kTremAmp			init 	0
	kTremAmp			chnget 	"tremAmp2"

	kTremFreq			init	0
	kTremFreq			chnget 	"tremFreq2"
	
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
	kAmp	expseg	iamp,	idur*idecaystartperc, 	iamp,	idur*(1-idecaystartperc)-0.01, 	0.001

	;   boundary conditions: 1 - clamped, 2- pivoting, 3 - free. When free is used on one end, the other must be clamped, otherwise the model will "blow up".
	;					LEFT      		| RIGHT     	| STIFFNESS | HIGH-      	|  SCANNING   | 30dB   		| STRIKE   | STRIKE   | WIDTH
	;					BOUNDARY  		| BOUNDARY  	|           | -FREQUENCY 	|  SPEED      | DECAY  		| POSITION | VELOCITY | OF
	;					CONDITION 		| CONDITION 	|           | LOSS       	|             | TIME   		|          |          | STRIKE
	ares	barmodel	kboundarycondl, kboundarycondr,    istiff,    ihighfreqloss,  kscanspeed,   idecay30db,     ipos,      ivel,      iwid

	kSrc = kAmp

	asrc = kSrc * ares * kVol * kExtDecay
	aleft, aright hrtfmove asrc, kAzimuth, kElevation, "hrtf\hrtf-44100-left.dat", "hrtf\hrtf-44100-right.dat"
	
;	printks "barmodel: %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t \n", 0.5, \\
;	kboundarycondl, kboundarycondr, istiff, ihighfreqloss, kbend, idecay30db, ipos, ivel, iwid

	out aleft, aright

endin

instr 3

	kVol				init	1 
	kVol				chnget 	"vol3"

	kAzimuth 			init 	0
	kAzimuth 			chnget 	"azimuth3"

	kElevation 			init 	0
	kElevation			chnget 	"elev3"

	idur				init	p3
	iamp				init	p4
	kboundarycondl 			init 	p5
	kboundarycondr 			init 	p6
	istiff				init 	p7 
	ihighfreqloss			init	p8
	kscanspeed			init	p9
	idecay30db			init	p10
	ivel				init	p11
	iwid 				init	p12
	idecaystartperc			init	p13 ; if 0 then no envelope
	idanger				init 	p14
	ielev				init  	p15
	ipos				init	0.5
	
	kExtDecay			init    1

	kMute				init 	1
	kMute				chnget	"mute3"

	kTremAmp			init 	0
	kTremAmp			chnget 	"tremAmp3"

	kTremFreq			init	0
	kTremFreq			chnget 	"tremFreq3"
	
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
	kAmp	expseg	iamp,	idur*idecaystartperc, 	iamp,	idur*(1-idecaystartperc)-0.01, 	0.001

	;   boundary conditions: 1 - clamped, 2- pivoting, 3 - free. When free is used on one end, the other must be clamped, otherwise the model will "blow up".
	;					LEFT      		| RIGHT     	| STIFFNESS | HIGH-      	|  SCANNING   | 30dB   		| STRIKE   | STRIKE   | WIDTH
	;					BOUNDARY  		| BOUNDARY  	|           | -FREQUENCY 	|  SPEED      | DECAY  		| POSITION | VELOCITY | OF
	;					CONDITION 		| CONDITION 	|           | LOSS       	|             | TIME   		|          |          | STRIKE
	ares	barmodel	kboundarycondl, kboundarycondr,    istiff,    ihighfreqloss,  kscanspeed,   idecay30db,     ipos,      ivel,      iwid

	kSrc = kAmp

	asrc = kSrc * ares * kVol * kExtDecay
	aleft, aright hrtfmove asrc, kAzimuth, kElevation, "hrtf\hrtf-44100-left.dat", "hrtf\hrtf-44100-right.dat"
	
;	printks "barmodel: %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t \n", 0.5, \\
;	kboundarycondl, kboundarycondr, istiff, ihighfreqloss, kbend, idecay30db, ipos, ivel, iwid

	out aleft, aright

endin

instr 4

	kVol				init	1 
	kVol				chnget 	"vol4"

	kAzimuth 			init 	0
	kAzimuth 			chnget 	"azimuth4"

	kElevation 			init 	0
	kElevation			chnget 	"elev4"

	idur				init	p3
	iamp				init	p4
	kboundarycondl 			init 	p5
	kboundarycondr 			init 	p6
	istiff				init 	p7 
	ihighfreqloss			init	p8
	kscanspeed			init	p9
	idecay30db			init	p10
	ivel				init	p11
	iwid 				init	p12
	idecaystartperc			init	p13 ; if 0 then no envelope
	idanger				init 	p14
	ielev				init  	p15
	ipos				init	0.5
	
	kExtDecay			init    1

	kMute				init 	1
	kMute				chnget	"mute4"

	kTremAmp			init 	0
	kTremAmp			chnget 	"tremAmp4"

	kTremFreq			init	0
	kTremFreq			chnget 	"tremFreq4"
	
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
	kAmp	expseg	iamp,	idur*idecaystartperc, 	iamp,	idur*(1-idecaystartperc)-0.01, 	0.001

	;   boundary conditions: 1 - clamped, 2- pivoting, 3 - free. When free is used on one end, the other must be clamped, otherwise the model will "blow up".
	;					LEFT      		| RIGHT     	| STIFFNESS | HIGH-      	|  SCANNING   | 30dB   		| STRIKE   | STRIKE   | WIDTH
	;					BOUNDARY  		| BOUNDARY  	|           | -FREQUENCY 	|  SPEED      | DECAY  		| POSITION | VELOCITY | OF
	;					CONDITION 		| CONDITION 	|           | LOSS       	|             | TIME   		|          |          | STRIKE
	ares	barmodel	kboundarycondl, kboundarycondr,    istiff,    ihighfreqloss,  kscanspeed,   idecay30db,     ipos,      ivel,      iwid

	kSrc = kAmp

	asrc = kSrc * ares * kVol * kExtDecay
	aleft, aright hrtfmove asrc, kAzimuth, kElevation, "hrtf\hrtf-44100-left.dat", "hrtf\hrtf-44100-right.dat"
	
;	printks "barmodel: %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t \n", 0.5, \\
;	kboundarycondl, kboundarycondr, istiff, ihighfreqloss, kbend, idecay30db, ipos, ivel, iwid

	out aleft, aright

endin

instr 5

	kVol				init	1 
	kVol				chnget 	"vol5"

	kAzimuth 			init 	0
	kAzimuth 			chnget 	"azimuth5"

	kElevation 			init 	0
	kElevation			chnget 	"elev5"

	idur				init	p3
	iamp				init	p4
	kboundarycondl 			init 	p5
	kboundarycondr 			init 	p6
	istiff				init 	p7 
	ihighfreqloss			init	p8
	kscanspeed			init	p9
	idecay30db			init	p10
	ivel				init	p11
	iwid 				init	p12
	idecaystartperc			init	p13 ; if 0 then no envelope
	idanger				init 	p14
	ielev				init  	p15
	ipos				init	0.5
	
	kExtDecay			init    1

	kMute				init 	1
	kMute				chnget	"mute5"

	kTremAmp			init 	0
	kTremAmp			chnget 	"tremAmp5"

	kTremFreq			init	0
	kTremFreq			chnget 	"tremFreq5"
	
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
	kAmp	expseg	iamp,	idur*idecaystartperc, 	iamp,	idur*(1-idecaystartperc)-0.01, 	0.001

	;   boundary conditions: 1 - clamped, 2- pivoting, 3 - free. When free is used on one end, the other must be clamped, otherwise the model will "blow up".
	;					LEFT      		| RIGHT     	| STIFFNESS | HIGH-      	|  SCANNING   | 30dB   		| STRIKE   | STRIKE   | WIDTH
	;					BOUNDARY  		| BOUNDARY  	|           | -FREQUENCY 	|  SPEED      | DECAY  		| POSITION | VELOCITY | OF
	;					CONDITION 		| CONDITION 	|           | LOSS       	|             | TIME   		|          |          | STRIKE
	ares	barmodel	kboundarycondl, kboundarycondr,    istiff,    ihighfreqloss,  kscanspeed,   idecay30db,     ipos,      ivel,      iwid

	kSrc = kAmp

	asrc = kSrc * ares * kVol * kExtDecay
	aleft, aright hrtfmove asrc, kAzimuth, kElevation, "hrtf\hrtf-44100-left.dat", "hrtf\hrtf-44100-right.dat"
	
;	printks "barmodel: %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t \n", 0.5, \\
;	kboundarycondl, kboundarycondr, istiff, ihighfreqloss, kbend, idecay30db, ipos, ivel, iwid

	out aleft, aright

endin

instr 6

	kVol				init	1 
	kVol				chnget 	"vol6"

	kAzimuth 			init 	0
	kAzimuth 			chnget 	"azimuth6"

	kElevation 			init 	0
	kElevation			chnget 	"elev6"

	idur				init	p3
	iamp				init	p4
	kboundarycondl 			init 	p5
	kboundarycondr 			init 	p6
	istiff				init 	p7 
	ihighfreqloss			init	p8
	kscanspeed			init	p9
	idecay30db			init	p10
	ivel				init	p11
	iwid 				init	p12
	idecaystartperc			init	p13 ; if 0 then no envelope
	idanger				init 	p14
	ielev				init  	p15
	ipos				init	0.5
	
	kExtDecay			init    1

	kMute				init 	1
	kMute				chnget	"mute6"

	kTremAmp			init 	0
	kTremAmp			chnget 	"tremAmp6"

	kTremFreq			init	0
	kTremFreq			chnget 	"tremFreq6"
	
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
	kAmp	expseg	iamp,	idur*idecaystartperc, 	iamp,	idur*(1-idecaystartperc)-0.01, 	0.001

	;   boundary conditions: 1 - clamped, 2- pivoting, 3 - free. When free is used on one end, the other must be clamped, otherwise the model will "blow up".
	;					LEFT      		| RIGHT     	| STIFFNESS | HIGH-      	|  SCANNING   | 30dB   		| STRIKE   | STRIKE   | WIDTH
	;					BOUNDARY  		| BOUNDARY  	|           | -FREQUENCY 	|  SPEED      | DECAY  		| POSITION | VELOCITY | OF
	;					CONDITION 		| CONDITION 	|           | LOSS       	|             | TIME   		|          |          | STRIKE
	ares	barmodel	kboundarycondl, kboundarycondr,    istiff,    ihighfreqloss,  kscanspeed,   idecay30db,     ipos,      ivel,      iwid

	kSrc = kAmp

	asrc = kSrc * ares * kVol * kExtDecay
	aleft, aright hrtfmove asrc, kAzimuth, kElevation, "hrtf\hrtf-44100-left.dat", "hrtf\hrtf-44100-right.dat"
	
;	printks "barmodel: %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t \n", 0.5, \\
;	kboundarycondl, kboundarycondr, istiff, ihighfreqloss, kbend, idecay30db, ipos, ivel, iwid

	out aleft, aright

endin

instr 7

	kVol				init	1 
	kVol				chnget 	"vol7"

	kAzimuth 			init 	0
	kAzimuth 			chnget 	"azimuth7"

	kElevation 			init 	0
	kElevation			chnget 	"elev7"

	idur				init	p3
	iamp				init	p4
	kboundarycondl 			init 	p5
	kboundarycondr 			init 	p6
	istiff				init 	p7 
	ihighfreqloss			init	p8
	kscanspeed			init	p9
	idecay30db			init	p10
	ivel				init	p11
	iwid 				init	p12
	idecaystartperc			init	p13 ; if 0 then no envelope
	idanger				init 	p14
	ielev				init  	p15
	ipos				init	0.5
	
	kExtDecay			init    1

	kMute				init 	1
	kMute				chnget	"mute7"

	kTremAmp			init 	0
	kTremAmp			chnget 	"tremAmp7"

	kTremFreq			init	0
	kTremFreq			chnget 	"tremFreq7"
	
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
	kAmp	expseg	iamp,	idur*idecaystartperc, 	iamp,	idur*(1-idecaystartperc)-0.01, 	0.001

	;   boundary conditions: 1 - clamped, 2- pivoting, 3 - free. When free is used on one end, the other must be clamped, otherwise the model will "blow up".
	;					LEFT      		| RIGHT     	| STIFFNESS | HIGH-      	|  SCANNING   | 30dB   		| STRIKE   | STRIKE   | WIDTH
	;					BOUNDARY  		| BOUNDARY  	|           | -FREQUENCY 	|  SPEED      | DECAY  		| POSITION | VELOCITY | OF
	;					CONDITION 		| CONDITION 	|           | LOSS       	|             | TIME   		|          |          | STRIKE
	ares	barmodel	kboundarycondl, kboundarycondr,    istiff,    ihighfreqloss,  kscanspeed,   idecay30db,     ipos,      ivel,      iwid

	kSrc = kAmp

	asrc = kSrc * ares * kVol * kExtDecay
	aleft, aright hrtfmove asrc, kAzimuth, kElevation, "hrtf\hrtf-44100-left.dat", "hrtf\hrtf-44100-right.dat"
	
;	printks "barmodel: %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t \n", 0.5, \\
;	kboundarycondl, kboundarycondr, istiff, ihighfreqloss, kbend, idecay30db, ipos, ivel, iwid

	out aleft, aright

endin




instr 31

	kVol				init	1 
	kVol				chnget 	"vol31"

	kAzimuth 			init 	0
	kAzimuth 			chnget 	"azimuth31"

	kElevation 			init 	0
	kElevation			chnget 	"elev31"

	iVol				init	p4
	iMode				init	p5

	asig loscil 1, 1, p5, 1, 0

	aleft, aright hrtfmove iVol * kVol * asig, kAzimuth, kElevation, "hrtf\hrtf-44100-left.dat", "hrtf\hrtf-44100-right.dat"

	out aleft, aright

endin

instr 32

	kVol				init	1 
	kVol				chnget 	"vol32"

	kAzimuth 			init 	0
	kAzimuth 			chnget 	"azimuth32"

	kElevation 			init 	0
	kElevation			chnget 	"elev32"

	iVol				init	p4
	iMode				init	p5

	asig loscil 1, 1, p5, 1, 0

	aleft, aright hrtfmove iVol * kVol * asig, kAzimuth, kElevation, "hrtf\hrtf-44100-left.dat", "hrtf\hrtf-44100-right.dat"

	out aleft, aright

endin

instr 33

	kVol				init	1 
	kVol				chnget 	"vol33"

	kAzimuth 			init 	0
	kAzimuth 			chnget 	"azimuth33"

	kElevation 			init 	0
	kElevation			chnget 	"elev33"

	iVol				init	p4
	iMode				init	p5

	asig loscil 1, 1, p5, 1, 0

	aleft, aright hrtfmove iVol * kVol * asig, kAzimuth, kElevation, "hrtf\hrtf-44100-left.dat", "hrtf\hrtf-44100-right.dat"

	out aleft, aright

endin

</CsInstruments>
<CsScore>
f 30 0 0 1 "marker-percuss01.wav" 0 0 0
f 31 0 0 1 "marker-percuss02.wav" 0 0 0
f 32 0 0 1 "marker-percuss03.wav" 0 0 0
i1 0 10000 0.1  1 1 500 8.03 0.01 0.5 2000 0.5 0.9 0 0 
</CsScore>
</CsoundSynthesizer>
