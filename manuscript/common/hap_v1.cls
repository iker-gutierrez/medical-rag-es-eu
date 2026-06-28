\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hap}
              [2002/06/10 v1
                HAP LaTeX document class]

\DeclareOption*{\PassOptionToClass{\CurrentOption}{article}}
% Execute the code of the options:
\ProcessOptions
\LoadClass[12pt,twoside,openwight,a4paper]{article}

% Some includes and definitions
%\usepackage{epsfig}
\usepackage{a4wide}
\usepackage{graphicx}
\usepackage{lastpage}
\usepackage{fancyhdr}
\usepackage{natbib}
\usepackage{color}

\newcommand{\HRule}{\rule{\linewidth}{0.5mm}}

\newcommand{\DocTitle}[1]   {\renewcommand{\DH@title}{#1}}
\newcommand{\DocDate}[1]    {\renewcommand{\DH@date}{#1}}
\newcommand{\DocNumber}[1]  {\renewcommand{\DH@number}{#1}}
\newcommand{\DocVersion}[1] {\renewcommand{\DH@version}{#1}}
\newcommand{\DH@title}{}
\newcommand{\DH@date}{}
\newcommand{\DH@number}{}
\newcommand{\DH@version}{}
\@onlypreamble{\DocTitle}
\@onlypreamble{\DocDate}
\@onlypreamble{\DocNumber}
\@onlypreamble{\DocVersion}

\setlength{\headheight}{3\baselineskip}
\setlength{\headsep}{3em}

\fancyhf{}
\rhead{\thepage/\pageref{LastPage}}
\lfoot{\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\\
Language Analysis and Processing}
%\rfoot{\DATE}
\lhead{\DH@title}

%\newcommand{\myetal}{\emph{et~al.}}
\bibliographystyle{common/hap}

\pagestyle{fancy}
% \let\DH@section=\section
% \renewcommand{\thepage}{\thesection-\arabic{page}}
% \renewcommand{\section}{\newpage\setcounter{page}{1}\DH@section}
