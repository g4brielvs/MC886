# A Multi-Layer Modeling Approach to Music Genre Classification

##
*Machine Learning project of MC886 at Unicamp*

Listening to music and perceiving its structure is a fairly easy task for
humans beings, even for listeners without formal musical training. However,
building computational models to mimic those processes is a hard problem.
The research is motivated by a large class of challenging applications in the media
business that require efficient and robust audio segmentation and
classification.
Application scenarios include audio streaming services and Web stream analysis,
automatic media monitoring, content- and descriptor-based search in large
multimedia databases, improving recommendation algorithms and playlists
suggestions, and artistic explorations.

## Goals and Objectives

- detect segment boundaries and transitions between musical sections on classical piano pieces.
- test the performance of the algorithm for automatic segmentation of audio streams into coherent or otherwise meaningful units corresponding to musical sections of more generic pieces, such as Concerti and Symphonies.

## Project Plan

The International Music Score Library Project ([IMSPL](http://imslp.org)) currently
comprises more than 288,000 scores and more than 31,000 public domain recordings.
Collections of piano sonatas by different composers (e.g. Haydn, Mozart,
Beethoven and Schubert) and/or different interpreters are going to be selected
and labeled accordingly to their corresponding musical sections. Audio descriptors algorithms
from [Essentia](http://essentia.upf.edu) (and possibly other available libraries) will populate the feature vectors.
Essentia is an open-source C++ library for audio analysis and audio-based music
information retrieval. This combination will give life to our training
dataset.

 We are interested in:

- **Tonal descriptors** notes, chords, keys, inharmonicity and dissonance
- **Rhythm descriptors** beat detection, BPM, rhythm transform, beat loudness, rhythm similarity or homogeneity.
- **Time-domain descriptors** duration, loudness, zero-crossing-rate log attack time and other signal envelope descriptors.
- **Statistical descriptors** median, mean, variance, power means, flatness.

## Reports

- [Final report](latex/report/ReportML.pdf)
- [Presentation](latex/presentation/ProjectML.pdf)
