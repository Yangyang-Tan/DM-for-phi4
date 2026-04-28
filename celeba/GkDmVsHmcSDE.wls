#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Pair-wise DM vs HMC propagator comparison for the SDE (em) sampler.
   4 directions x {log-log, log-linear} = 8 PDFs.
   DM uses the 10k-sample reference at ep=9999; HMC is the full
   CelebA-128 training dataset. *)

SetDirectory[DirectoryName[If[$InputFileName === "", NotebookFileName[], $InputFileName]]];
Get[FileNameJoin[{"..", "academic.m"}]];
Needs["MaTeX`"];

corrDir   = "celeba_128_ncsnpp/correlation";
figureDir = "celeba_128_ncsnpp/figures";

epochStr        = "9999";    (* on-disk file tag: Lightning saves at end of epoch N=9999 *)
epochDisplayStr = "10000";   (* = N+1 completed training epochs; used in labels and output paths *)
sampler         = "em";       (* SDE only for this script *)

dirPretty[d_] := Switch[d,
  "x", "X", "y", "Y", "diagonal", "Diagonal", "all", "All", _, d];

(* "all" and the axial directions each have their own suffix *)
dirSuffix[d_] := "_" <> d;

(* Load a .dat file, drop k=0 row and attach Around[.] uncertainty *)
loadWithErr[path_] := Module[{rows},
  rows = Select[Import[path, "Table"], #[[1]] > 0 &];
  {#[[1]], Around[#[[2]], #[[3]]]} & /@ rows];

(* ------------------------------------------------------------
   Build each panel; `scaleKey` selects log-log ("LogLog") or
   log-linear ("LogLinear", x-log, y-linear).
   ------------------------------------------------------------ *)

maTeXLabel[lbl_, size_:14] :=
  MaTeX[lbl, ContentPadding -> False, FontSize -> size];

makePanel[direction_String, scaleKey_String] :=
 Module[{dirSfx, trainFile, dmFile, dmData, trData, plot, scaleSpec, yLabel},
  dirSfx    = dirSuffix[direction];
  trainFile = FileNameJoin[{corrDir, "G_k_train_celeba_128" <> dirSfx <> ".dat"}];
  dmFile    = FileNameJoin[{corrDir,
    "G_k_DM_celeba_128_" <> sampler <> dirSfx <> "_epoch=" <> epochStr <> ".dat"}];
  dmData    = loadWithErr[dmFile];
  trData    = loadWithErr[trainFile];

  scaleSpec = Switch[scaleKey,
    "LogLog",    {"Log", "Log"},
    "LogLinear", {"Log", Automatic}];

  plot = ListLinePlot[
    {dmData, trData},
    IntervalMarkers -> "Bars",
    IntervalMarkersStyle -> AbsoluteThickness[1.3],
    ScalingFunctions -> scaleSpec,
    PlotRange -> All,
    PlotStyle -> {Automatic,
                  Directive[Black, Dashed]},
    PlotLegends -> Placed[
      LineLegend[
        {maTeXLabel["\\text{DM (SDE, ep}=" <> epochDisplayStr <> "\\text{)}", 13],
         maTeXLabel["\\text{CelebA HMC}", 13]},
        LegendMarkerSize -> {36, 10},
        LegendLayout -> {"Column", 1}],
      If[scaleKey === "LogLog", Scaled[{0.22, 0.22}], Scaled[{0.78, 0.72}]]],
    Epilog -> {Inset[
        maTeXLabel["\\text{" <> dirPretty[direction] <>
                   " / } \\mathrm{SDE}", 15],
        Scaled[{0.78, 0.92}], {Right, Top}]},
    FrameLabel -> {maTeXLabel["|k|", 15],
                   maTeXLabel["G(|k|)", 15]},
    ImageSize -> 460];
  plot];

Do[
  panel  = makePanel[direction, "LogLog"];
  fname  = "prop_celeba_128_em_" <> direction <> "_ep10000_loglog.pdf";
  Export[FileNameJoin[{figureDir, fname}], panel];
  Print["saved: ", fname];
  ,
  {direction, {"x", "y", "diagonal", "all"}}];

Do[
  panel  = makePanel[direction, "LogLinear"];
  fname  = "prop_celeba_128_em_" <> direction <> "_ep10000_loglin.pdf";
  Export[FileNameJoin[{figureDir, fname}], panel];
  Print["saved: ", fname];
  ,
  {direction, {"x", "y", "diagonal", "all"}}];

(* Convenience composite 2x4 (rows = scale, cols = direction) for quick inspection *)
composite = Grid[{
  Table[makePanel[d, "LogLog"],    {d, {"x", "y", "diagonal", "all"}}],
  Table[makePanel[d, "LogLinear"], {d, {"x", "y", "diagonal", "all"}}]},
  Spacings -> {1.5, 1.5}, Frame -> None];

Export[FileNameJoin[{figureDir, "prop_celeba_128_em_ep10000_DMvsHMC_4x2.pdf"}],
       composite];
Print["saved composite: prop_celeba_128_em_ep10000_DMvsHMC_4x2.pdf"];
