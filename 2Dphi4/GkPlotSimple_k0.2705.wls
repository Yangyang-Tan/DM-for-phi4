#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Single clean DM vs HMC propagator comparison on log-log scale.
   2 curves: DM (em, ep=10000, N=2048) and HMC (training, wolff+FAHMC).
   Output: one PDF in sigma_comparison_L128/figures/. *)

SetDirectory[DirectoryName[If[$InputFileName === "", NotebookFileName[], $InputFileName]]];
Get[FileNameJoin[{"..", "academic.m"}]];
Needs["MaTeX`"];

k      = "0.2705";
sigma  = "sigma450";
ep     = "10000";
method = "em";

dd = "sigma_comparison_L128/L128_k" <> k <> "_" <> sigma <> "/dat_cache/";
fd = "sigma_comparison_L128/figures/";
If[!DirectoryQ[fd], CreateDirectory[fd]];

loadRaw[f_] := Select[Import[f], (#[[1]] > 0 && #[[2]] > 0) &];
toAround[rows_] := {#1, Around[#2, #3]} & @@@ rows;

dmFile    = FileNameJoin[{dd, "G_k_DM_" <> method <> "_epoch=" <> ep <> ".dat"}];
trainFile = FileNameJoin[{dd, "G_k_train.dat"}];

dmRaw    = loadRaw[dmFile];
trainRaw = loadRaw[trainFile];

(* Central values only (no error bars) for log-log. *)
dmCenter    = {#[[1]], #[[2]]} & /@ dmRaw;
trainCenter = {#[[1]], #[[2]]} & /@ trainRaw;

(* Around (with error bars) for log-linear. *)
dmData    = toAround[dmRaw];
trainData = toAround[trainRaw];

styles = {Directive[Black, Dashed, Thickness[0.003]],
          Directive[RGBColor[0.85, 0.15, 0.15], Thickness[0.003]]};

legendItems = {
  MaTeX["\\text{HMC (wolff+FAHMC)}", ContentPadding -> False, FontSize -> 15],
  MaTeX["\\text{DM (SDE, ep=10000)}", ContentPadding -> False, FontSize -> 15]};

annot = MaTeX["\\kappa = 0.2705,\\; L = 128,\\; \\lambda = 0.022",
              ContentPadding -> False, FontSize -> 15];

(* Figure 1: log-log, NO error bars *)
plotLogLog = ListLinePlot[
  {trainCenter, dmCenter},
  ScalingFunctions -> {"Log", "Log"},
  PlotRange -> All,
  PlotStyle -> styles,
  PlotLegends -> Placed[
    LineLegend[legendItems,
      LegendMarkerSize -> {40, 10},
      LegendLayout -> {"Column", 1}],
    Scaled[{0.25, 0.25}]],
  Epilog -> {Inset[annot, Scaled[{0.97, 0.92}], {Right, Top}]},
  FrameLabel -> {MaTeX["|k|", ContentPadding -> False, FontSize -> 16],
                 MaTeX["G(|k|)", ContentPadding -> False, FontSize -> 16]},
  ImageSize -> 500];
fname1 = "prop_DM_vs_HMC_phi4_L128_k0.2705_loglog.pdf";
Export[FileNameJoin[{fd, fname1}], plotLogLog];
Print["saved: ", fd <> fname1];

(* Figure 2: log-linear (x log, y linear), with error bars *)
plotLogLin = ListLinePlot[
  {trainData, dmData},
  IntervalMarkers -> "Fences",
  IntervalMarkersStyle -> <|"FenceWidth" -> 0.06, "FenceStyle" -> Thickness[0.003]|>,
  ScalingFunctions -> {"Log", Automatic},
  PlotRange -> All,
  PlotStyle -> styles,
  PlotLegends -> Placed[
    LineLegend[legendItems,
      LegendMarkerSize -> {40, 10},
      LegendLayout -> {"Column", 1}],
    Scaled[{0.78, 0.70}]],
  Epilog -> {Inset[annot, Scaled[{0.97, 0.92}], {Right, Top}]},
  FrameLabel -> {MaTeX["|k|", ContentPadding -> False, FontSize -> 16],
                 MaTeX["G(|k|)", ContentPadding -> False, FontSize -> 16]},
  ImageSize -> 500];
fname2 = "prop_DM_vs_HMC_phi4_L128_k0.2705_loglin.pdf";
Export[FileNameJoin[{fd, fname2}], plotLogLin];
Print["saved: ", fd <> fname2];
