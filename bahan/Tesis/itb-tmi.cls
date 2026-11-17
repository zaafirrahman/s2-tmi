\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{itb-tmi}[2026/11/13 Tesis Magister ITB (juknis SPs Mei 2019)]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass[12pt,a4paper]{report}
\RequirePackage[T1]{fontenc}
\RequirePackage{tgtermes} % setara Times New Roman
\RequirePackage[english,indonesian]{babel}
\RequirePackage{geometry}
\if@twoside
 \geometry{a4paper,inner=4cm,outer=3cm,top=3cm,bottom=3cm,nohead,footskip=1.5cm}
\else
 \geometry{a4paper,left=4cm,right=3cm,top=3cm,bottom=3cm,nohead,footskip=1.5cm}
\fi
\RequirePackage{setspace,graphicx,titlesec,tocloft,caption,array,xcolor}
\onehalfspacing
\setlength{\parindent}{0pt}\setlength{\parskip}{12pt}
\renewcommand{\thechapter}{\Roman{chapter}}
\renewcommand{\thesection}{\thechapter.\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\thefigure}{\thechapter.\arabic{figure}}
\renewcommand{\thetable}{\thechapter.\arabic{table}}
\renewcommand{\theequation}{\thechapter.\arabic{equation}}
\titleformat{\chapter}[block]{\centering\bfseries\fontsize{14}{17}\selectfont}{Bab \thechapter}{1em}{}
\titlespacing*{\chapter}{0pt}{-33pt}{12pt} % kompensasi \parskip dan \topskip agar judul bab sejajar dengan judul halaman awal
\titleformat{\section}[hang]{\bfseries\normalsize}{\thesection}{1em}{}
\titleformat{\subsection}[hang]{\bfseries\normalsize}{\thesubsection}{1em}{}
\titlespacing*{\section}{0pt}{6pt}{0pt}\titlespacing*{\subsection}{0pt}{6pt}{0pt}
\captionsetup{labelsep=space,format=hang,justification=justified,skip=6pt}
\renewcommand{\cftchappresnum}{Bab }\setlength{\cftchapnumwidth}{3.6em}
\renewcommand{\cftchapfont}{\normalfont}\renewcommand{\cftchappagefont}{\normalfont}
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
\setlength{\cftbeforechapskip}{12pt}\setlength{\cftsecindent}{3.6em}\setlength{\cftsecnumwidth}{3em}
\setlength{\cftsubsecindent}{6.6em}\setlength{\cftsubsecnumwidth}{3.6em}
\renewcommand{\cftfigpresnum}{Gambar }\setlength{\cftfignumwidth}{5.5em}\setlength{\cftfigindent}{0pt}
\renewcommand{\cfttabpresnum}{Tabel }\setlength{\cfttabnumwidth}{5.5em}\setlength{\cfttabindent}{0pt}
\renewcommand{\cfttoctitlefont}{\hfill\bfseries\fontsize{14}{17}\selectfont}\renewcommand{\cftaftertoctitle}{\hfill}
\renewcommand{\cftloftitlefont}{\hfill\bfseries\fontsize{14}{17}\selectfont}\renewcommand{\cftafterloftitle}{\hfill}
\renewcommand{\cftlottitlefont}{\hfill\bfseries\fontsize{14}{17}\selectfont}\renewcommand{\cftafterlottitle}{\hfill}
\AtBeginDocument{\renewcommand{\contentsname}{DAFTAR ISI}\renewcommand{\listfigurename}{DAFTAR GAMBAR DAN ILUSTRASI}\renewcommand{\listtablename}{DAFTAR TABEL}}
% judul halaman awal (14 pt tebal, tengah) + masuk daftar isi
\newcommand{\judulawal}[1]{\clearpage\addcontentsline{toc}{chapter}{#1}\begin{center}\bfseries\fontsize{14}{17}\selectfont #1\end{center}\vspace{-6pt}}
% daftar-daftar tanpa judul bawaan, supaya sama dengan halaman awal lain
\newlistof{lampiran}{lam}{DAFTAR LAMPIRAN}
\setlength{\cftlampiranindent}{0pt}\setlength{\cftlampirannumwidth}{5.5em}
\newcommand{\daftarisi}{\judulawal{DAFTAR ISI}\@starttoc{toc}}
\newcommand{\daftarlampiran}{\judulawal{DAFTAR LAMPIRAN}\@starttoc{lam}}
\newcommand{\daftargambar}{\judulawal{DAFTAR GAMBAR DAN ILUSTRASI}\@starttoc{lof}}
\newcommand{\daftartabel}{\judulawal{DAFTAR TABEL}\@starttoc{lot}}
% lampiran: halaman pemisah, lalu "Lampiran A  Judul" rata kiri
\newcommand{\mulailampiran}{\clearpage\thispagestyle{plain}\vspace*{\fill}\begin{center}\bfseries\fontsize{14}{17}\selectfont LAMPIRAN\end{center}\vspace*{\fill}\addcontentsline{toc}{chapter}{LAMPIRAN}\appendix}
\newcommand{\lampiranbaru}[1]{\clearpage\refstepcounter{chapter}\noindent{\bfseries Lampiran \thechapter\hspace{1em}#1}\par\addcontentsline{lam}{lampiran}{\protect\numberline{Lampiran \thechapter}#1}}
\endinput
