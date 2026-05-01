#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Log-log DM vs HMC with overlayed power-law reference curves.
   Style follows user's template: scatter+line (no fences), HMC black dashed,
   two theoretical asymptotes: 1/p^{2-η} (η=0.25, 2D Ising class) and 1/p^2.
   Near-critical κ=0.2705, L=128, ep=10000 em (SDE), N=2048. *)

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

dmFile    = FileNameJoin[{dd, "G_k_DM_" <> method <> "_epoch=" <> ep <> ".dat"}];
trainFile = FileNameJoin[{dd, "G_k_train.dat"}];

(* load: {|k|, G, G_err}, keep k>0, trim first+last bin (visual noise) *)
loadData[f_] := Module[{rows},
   rows = Select[Import[f], (#[[1]] > 0) &];
   ({#1, Around[#2, #3]} & @@@ rows)[[2 ;; -2]]];

dmData    = loadData[dmFile];
trainData = loadData[trainFile];

(* Power-law amplitudes tuned for this dataset by matching at |k|≈1.
   Adjust if the visual vertical offset is off. *)
ampEta  = 1.94;   (* 1/p^{2-η}, η=0.25 (2D Ising class) *)
ampG    = 1.85;   (* 1/p^2     (Gaussian / free-field) *)

dataPlot = ListLinePlot[
  {dmData, trainData},
  ScalingFunctions -> {"Log", "Log"},
  PlotStyle -> {
    Directive[RGBColor[0.85, 0.15, 0.15], Thickness[0.003]],      (* DM *)
    Directive[Black, Dashed, Thickness[0.003]]                    (* HMC *)
  },
  PlotLegends -> Placed[
    LineLegend[
      MaTeX[#, ContentPadding -> False, FontSize -> 15] & /@
        {"\\mathrm{DM}", "\\mathrm{HMC}"},
      LegendMarkerSize -> {40, 12},
      LegendLayout -> {"Column", 1}],
    Scaled[{0.28, 0.35}]],
  FrameLabel -> {
    MaTeX["|k|", ContentPadding -> False, FontSize -> 16],
    MaTeX["G(|k|)", ContentPadding -> False, FontSize -> 16]},
  PlotRange -> All,
  ImageSize -> 500];

refPlot = Plot[
  {ampEta / p^(2 - 0.25), ampG / p^2},
  {p, 0.05, 5.0},
  ScalingFunctions -> {"Log", "Log"},
  PlotStyle -> {
    Directive[RGBColor[0.20, 0.40, 0.70], Thin, Dotted, Thickness[0.002]],
    Directive[RGBColor[0.20, 0.60, 0.30], Thin, DotDashed, Thickness[0.002]]
  },
  PlotLegends -> Placed[
    LineLegend[
      MaTeX[#, ContentPadding -> False, FontSize -> 15] & /@
        {"\\propto 1/p^{\\,2-\\eta}\\;(\\eta=0.25)",
         "\\propto 1/p^{\\,2}"},
      LegendMarkerSize -> {40, 12},
      LegendLayout -> {"Column", 1}],
    Scaled[{0.80, 0.80}]]];

fig = Show[
  dataPlot, refPlot,
  Epilog -> {Inset[
     MaTeX["\\kappa = 0.2705,\\; L = 128,\\; \\lambda = 0.022",
           ContentPadding -> False, FontSize -> 15],
     Scaled[{0.97, 0.92}], {Right, Top}]},
  ImageSize -> 520];

fname = "prop_DM_vs_HMC_phi4_L128_k0.2705_loglog_refs.pdf";
Export[FileNameJoin[{fd, fname}], fig];
Print["saved: ", fd <> fname];
