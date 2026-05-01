#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Cumulants kappa_{2,4,6,8} vs training epoch for 3D phi^4 L=64 k=0.2
   (sigma=2760 run). DM samples = EM, 2000 steps, log-schedule; reference =
   Wolff+FAHMC training data (identical physics: k=0.2, l=0.9, 64^3).

   Input:
     phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma2760/cumulants/cumulants_dm_em.dat
     phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma2760/cumulants/cumulants_hmc.dat

   Output: 4 individual PDFs + 1 composite 2x2 PDF under
     phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma2760/figures/

   Each panel: one line = |kappa_n^DM|(epoch) with bootstrap fences;
               one point = |kappa_n^HMC| placed at x = last_epoch. *)

SetDirectory[DirectoryName[If[$InputFileName === "", NotebookFileName[], $InputFileName]]];
Get[FileNameJoin[{"..", "academic.m"}]];
Needs["MaTeX`"];

(* ---- configuration ---------------------------------------------------- *)
ckptDir = "phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma2760";
dataDir = FileNameJoin[{ckptDir, "cumulants"}];
figDir  = FileNameJoin[{ckptDir, "figures"}];
If[!DirectoryQ[figDir], CreateDirectory[figDir]];

orders   = {2, 4, 6, 8};
absLbl   = <|2 -> "|\\kappa_2|", 4 -> "|\\kappa_4|",
             6 -> "|\\kappa_6|", 8 -> "|\\kappa_8|"|>;

(* ---- load data -------------------------------------------------------- *)
dmRaw  = Import[FileNameJoin[{dataDir, "cumulants_dm_em.dat"}], "Table",
                "IgnoreEmptyLines" -> True];
dmRaw  = Select[dmRaw, (Length[#] >= 9 && NumberQ[#[[1]]]) &];
dmRaw  = SortBy[dmRaw, First];
epochs = dmRaw[[All, 1]];
xMin   = Min[epochs]; xMax = Max[epochs];

hmcRaw = Import[FileNameJoin[{dataDir, "cumulants_hmc.dat"}], "Table",
                "IgnoreEmptyLines" -> True];
hmcRaw = Select[hmcRaw, (Length[#] == 3 && NumberQ[#[[1]]]) &];
hmcAssoc = AssociationThread[hmcRaw[[All, 1]] -> hmcRaw[[All, {2, 3}]]];

(* DM rows for order k: {epoch, |kappa_n|, err}. File columns:
   1 epoch, 2 k2, 3 k2e, 4 k4, 5 k4e, 6 k6, 7 k6e, 8 k8, 9 k8e
   Because orders are exactly (2,4,6,8), central col = k, err col = k + 1. *)
dmRows[k_] := Module[{c = k},
  {#[[1]], Abs[#[[c]]], #[[c + 1]]} & /@ dmRaw];

(* {x, |kappa_n|, err} for HMC at x = xMax. *)
hmcPt[k_] := Module[{mu, er},
  {mu, er} = hmcAssoc[k];
  {xMax, Abs[mu], er}];

toAround[row_] := {row[[1]], Around[row[[2]], row[[3]]]};

(* ---- colors ----------------------------------------------------------- *)
dmColor  = RGBColor[0.192157, 0.388235, 0.807843]; (* blue *)
hmcColor = RGBColor[0.790588, 0.201176, 0.];       (* red  *)

(* ---- single-panel plot builder --------------------------------------- *)
Options[buildPanel] = {FontSize -> 15, Legend -> True, ImageSize -> 500};
buildPanel[k_, OptionsPattern[]] :=
 Module[{dm, hmc, dmAround, hmcAround, xLbl, yLbl, annot, legend,
         fontSz, legendQ, imSz, dmPlot, hmcPlot},
  fontSz  = OptionValue[FontSize];
  legendQ = OptionValue[Legend];
  imSz    = OptionValue[ImageSize];

  dm  = dmRows[k];
  hmc = hmcPt[k];
  (* Central values only — early epochs have bootstrap err comparable to |kappa|
     (sign flips), which breaks log-y fences. Keep the line clean. *)
  dmCenter = {#[[1]], #[[2]]} & /@ dm;
  hmcCenter = {{hmc[[1]], hmc[[2]]}};

  xLbl = MaTeX["\\text{epoch}", ContentPadding -> False, FontSize -> fontSz];
  yLbl = MaTeX[absLbl[k],       ContentPadding -> False, FontSize -> fontSz];
  annot = MaTeX[
     "\\kappa = 0.2,\\; L = 64,\\; \\sigma = 2760,\\; \\mathrm{SDE}",
     ContentPadding -> False, FontSize -> fontSz - 2];

  legend = If[legendQ,
    Placed[
      PointLegend[{
         Directive[dmColor, AbsoluteThickness[1.8]],
         Directive[hmcColor, PointSize[0.03]]},
        {MaTeX["\\mathrm{DM}\\;(\\mathrm{SDE})", ContentPadding -> False,
               FontSize -> fontSz - 2],
         MaTeX["\\mathrm{HMC}", ContentPadding -> False,
               FontSize -> fontSz - 2]},
        LegendMarkers -> {Automatic,
          Graphics[{hmcColor, Disk[]}]},
        LegendMarkerSize -> 12, LegendLayout -> {"Column", 1}],
      Scaled[{0.82, 0.82}]],
    None];

  dmPlot = ListLinePlot[dmCenter,
    ScalingFunctions -> {"Log", "Log"},
    PlotStyle -> Directive[dmColor, AbsoluteThickness[1.8]],
    PlotRange -> All,
    Mesh -> All, MeshStyle -> Directive[dmColor, PointSize[0.008]]];

  hmcPlot = ListPlot[hmcCenter,
    ScalingFunctions -> {"Log", "Log"},
    PlotStyle -> Directive[hmcColor, PointSize[0.028]],
    PlotMarkers -> {Graphics[{hmcColor, Disk[]}], 0.035}];

  combined = Show[dmPlot, hmcPlot,
    FrameLabel -> {xLbl, yLbl},
    PlotRange -> All,
    Epilog -> {Inset[annot, Scaled[{0.03, 0.05}], {Left, Bottom}]},
    ImageSize -> imSz];
  If[legendQ, Legended[combined, legend], combined]];

(* ---- individual PDFs -------------------------------------------------- *)
Do[
  plot  = buildPanel[k, FontSize -> 15, Legend -> True, ImageSize -> 500];
  fname = "cumulant_k" <> ToString[k] <>
          "_vs_epoch_3d_L64_k0.2_sigma2760_em.pdf";
  Export[FileNameJoin[{figDir, fname}], plot];
  Print["saved: ", figDir, "/", fname],
  {k, orders}];

(* ---- 2x2 composite ---------------------------------------------------- *)
composite = Grid[{
   {buildPanel[2, FontSize -> 13, Legend -> True,  ImageSize -> 380],
    buildPanel[4, FontSize -> 13, Legend -> False, ImageSize -> 380]},
   {buildPanel[6, FontSize -> 13, Legend -> False, ImageSize -> 380],
    buildPanel[8, FontSize -> 13, Legend -> False, ImageSize -> 380]}},
   Spacings -> {1.5, 1.5}, Frame -> None];
Export[FileNameJoin[{figDir,
   "cumulants_2x2_vs_epoch_3d_L64_k0.2_sigma2760_em.pdf"}], composite];
Print["saved: ", figDir,
      "/cumulants_2x2_vs_epoch_3d_L64_k0.2_sigma2760_em.pdf"];
