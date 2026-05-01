#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[NotebookDirectory[]];
Get["academic.m"];
Needs["MaTeX`"];


ListLinePlot[{{#1,Around[#2,#3]}&@@@Import["G_k_128_boot_DM_26_0.022_epoch=0099.dat"],(*{#1,Around[#2,#3]}&@@@Import["G_k_32_jk_DM_512_2_epoch=00_steps=0400.dat"],{#1,Around[#2,#3]}&@@@Import["G_k_32_jk_DM_512_2_epoch=01.dat"],*)(*{#1,Around[#2,#3]}&@@@Import["G_k_32_jk_DM_512_2_epoch=99.dat"],*){#1,Around[#2,#3]}&@@@Import["G_k_128_boot_DM_26_0.022_epoch=0999.dat"],{#1,Around[#2,#3]}&@@@Import["G_k_128_boot_DM_26_0.022_epoch=1199.dat"],{#1,Around[#2,#3]}&@@@Import["G_k_128_boot_DM_26_0.022_epoch=1299.dat"],{#1,Around[#2,#3]}&@@@Import["G_k_128_boot_DM_26_0.022_epoch=4999.dat"],{#1,Around[#2,#3]}&@@@Import["G_k_128_boot_HMC_26_0.022.dat"]}[[All,2;;-2,All]],IntervalMarkers->"Fences",IntervalMarkersStyle-><|"FenceWidth"->0.035|>
,ScalingFunctions->{"Log",Automatic},PlotRange->All,PlotStyle->{Automatic,Automatic,Automatic,Automatic,Automatic,Directive[Black,Dashed]},PlotLegends->Placed[LineLegend[(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"100~ \\mathrm{Epoches}","100~ \\mathrm{Epoches}","1000~ \\mathrm{Epoches}","1500~ \\mathrm{Epoches}","2000~ \\mathrm{Epoches}","\\mathrm{HMC}"}),LegendMarkerSize->{40,12},LegendLayout->{"Column",1}],Scaled[{0.75,0.7}]],FrameLabel->(MaTeX[#,ContentPadding->False,FontSize->15]&/@{"|k|","G(|k|)"})]


Export["prop_k=0.26_log.pdf",%]
