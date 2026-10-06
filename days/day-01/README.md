<div align="center">

# Day 01: Foundations of Reinforcement Learning

**Reinforcement Learning and Language Model Alignment (VTR UGE 21)**  
Prof. Dr. Utku Kose, Süleyman Demirel University

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkukose/rl-alignment-veltech-NB-lecture/blob/main/days/day-01/NB01_foundations_of_rl.ipynb) [![Interactive lab](https://img.shields.io/badge/interactive%20lab-open-0E7A78)](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lab.html) [![Lecture page](https://img.shields.io/badge/lecture%20page-open-1F5F8B)](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lecture.html) [![PDF notes](https://img.shields.io/badge/PDF%20notes-download-B97813)](Day01_Lecture_Notes.pdf)

</div>

Learning from interaction: bandits and exploration, Markov decision processes, values and the Bellman equation, and a small world that can be solved exactly. **Estimated study time:** 6 to 8 hours.

## Materials of the day

Each material has its own role. Start with the lecture page; the study path below gives the order and the time of each step.

| Material | What it holds | Open |
|---|---|---|
| Lecture page | The concepts of the day, with animations, knowledge checks, review cards and the references | [Lecture page](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lecture.html) |
| Colab notebook | Python step 1: Values, variables, lists and decisions, then the hands-on sections with exercises, an application switch and the application challenges | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkukose/rl-alignment-veltech-NB-lecture/blob/main/days/day-01/NB01_foundations_of_rl.ipynb) |
| Interactive lab | Foundations lab: Three practice parts, a self-assessment and an exportable learning log | [Interactive lab](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lab.html) |
| PDF lecture notes | The lecture and the Python step in one printable file | [PDF notes](Day01_Lecture_Notes.pdf) |

<table><tr><td width="50%"><a href="https://colab.research.google.com/github/utkukose/rl-alignment-veltech-NB-lecture/blob/main/days/day-01/NB01_foundations_of_rl.ipynb"><img src="screenshots/nb_1.png" alt="A figure from the notebook of day 1"></a><br><sub>From the Colab notebook</sub></td><td width="50%"><a href="https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lab.html"><img src="screenshots/lab.png" alt="Interactive lab of day 1"></a><br><sub>The interactive lab</sub></td></tr></table>

## Learning outcomes

By the end of the day, students are expected to describe the agent-environment interface and state three ways in which the reward hypothesis can fail, to implement and compare epsilon-greedy, UCB and Thompson sampling by cumulative regret over many seeds, to formulate sequence generation as a Markov decision process, to write the Bellman expectation and optimality equations and solve small processes with policy and value iteration, to obtain the exact optimum of TokenWorld as a reference for later days, and to report a stochastic experiment with seeds, spread and sample size. In Python, students are expected to use values, variables, lists, loops, decisions, functions and dictionaries to simulate and score a bandit.

## Study path

| Step | Activity | Suggested time |
|---|---|---|
| 1 | Read the [lecture page](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lecture.html) and answer its knowledge checks | 1 hour 30 minutes |
| 2 | Work through parts A, B and C of the [interactive lab](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lab.html) | 45 minutes |
| 3 | Run Python step 1: Values, variables, lists and decisions in the [Colab notebook](https://colab.research.google.com/github/utkukose/rl-alignment-veltech-NB-lecture/blob/main/days/day-01/NB01_foundations_of_rl.ipynb#scrollTo=python-step), right after the setup cell | 1 hour |
| 4 | Work through the numbered sections of the notebook and their exercises | 2 hours |
| 5 | Use the application switch of the notebook and solve the application challenge of your field | 45 minutes |
| 6 | Take the self-assessment in the [interactive lab](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lab.html) (tab: Check yourself) | 20 minutes |
| 7 | Write the reflection in the lab, export the learning log and complete the daily task below | 45 minutes |

## Live session plan

The day runs as one synchronous session in class or online, and each material has one role in it. The lecture page is on the screen, and at each orange Colab box the instructor shows the matching section in a copy of the notebook that has already been run. Students work in their own copy of the notebook from top to bottom, and each lab part follows the lecture part it practises. The PDF lecture notes serve reading after the session, and the study path above serves self-paced study.

| Time | Activity |
|---|---|
| 0:00 to 0:15 | Opening: Five slot machines, played by the class against an algorithm in lab Part A. A tour of the course page, then students open the notebook in Colab, save a copy in Drive and run section 0 |
| 0:15 to 0:50 | Lecture page, part 1: Learning from interaction, bandits with the regret animation, Markov decision processes; then lab Part B with the whole class |
| 0:50 to 1:15 | Colab, together: Python step 1, ending with the regret of a bandit computed by hand |
| 1:15 to 1:25 | Break |
| 1:25 to 2:00 | Lecture page, part 2: Values and the Bellman equation with the value-iteration animation, the exact optimum as a yardstick; then lab Part C by hand |
| 2:00 to 2:40 | Colab, in pairs: Sections 1 to 4 with their exercises, then section 5 with a different field for each pair |
| 2:40 to 3:00 | Closing: Seeds and spread, or why one run proves little, then the self-assessment of the lab; after the session: The PDF notes, the daily task and the optional section 6 |

## Daily task and submission

Run the bandit comparison of section 1 on a problem where the arms are far apart, for example win rates of 0.1, 0.3, 0.5, 0.7 and 0.9, and compare the ranking with the close arms of the lecture. Write about 200 words on which column of the table you would report for a clinical trial and which for a recommender system, and why.

The task is optional and supports self-learning and a personal portfolio. During an active delivery of the course, it can be sent together with the exported learning log to utkukose@sdu.edu.tr or utkukose@gmail.com.

## Research and report assignment (optional)

**Reproducibility in reinforcement learning.** Review the evidence on seed variance and reporting practices in reinforcement learning &#91;[6](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lecture.html#ref-6)&#93; and relate it to bandit theory &#91;[2](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lecture.html#ref-2), [3](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lecture.html#ref-3)&#93;. Propose a reporting checklist and apply it to two recent papers of your choice. The report should be about 1500 words with at least eight sources.
