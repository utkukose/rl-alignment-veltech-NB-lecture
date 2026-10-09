<div align="center">

# Reinforcement Learning and Language Model Alignment

**VTR UGE 21 Value Added Course, Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology, Chennai, India**

*A five-day course from Bellman equations to direct preference optimisation on one environment, with lecture pages, animations, interactive labs, Colab notebooks and a Python track for beginners*

![last update](https://img.shields.io/badge/last%20update-October%202026-B97813) ![course code](https://img.shields.io/badge/course%20code-VTR%20UGE%2021-1F5F8B) ![format](https://img.shields.io/badge/format-5%20days-1F5F8B) ![delivery](https://img.shields.io/badge/delivery-synchronous%20and%20asynchronous-0E7A78) ![notebooks](https://img.shields.io/badge/notebooks-Google%20Colab-F9AB00) ![content](https://img.shields.io/badge/content-CC%20BY%204.0-555555) ![code](https://img.shields.io/badge/code-MIT-555555)

**This course is updated in line with current developments in the field. Last update: October 2026.**

**Prof. Dr. Utku Kose**

Full Professor, Department of Computer Engineering, Süleyman Demirel University, Isparta, Türkiye  
Founding Director, AI Application and Research Center (YAZEM), Süleyman Demirel University  
Head of the Computer Science Division, Department of Computer Engineering, Süleyman Demirel University  
Additional affiliations: University of North Dakota (USA), Universidad Panamericana (Mexico City, Mexico), Vel Tech University (Chennai, India)  
IEEE Senior Member, ACM Professional Member

[ORCID 0000-0002-9652-6415](https://orcid.org/0000-0002-9652-6415) | [utkukose.com](https://www.utkukose.com) | [github.com/utkukose](https://github.com/utkukose)

[utkukose@sdu.edu.tr](mailto:utkukose@sdu.edu.tr) | [utku.kose@und.edu](mailto:utku.kose@und.edu) | [ukose@up.edu.mx](mailto:ukose@up.edu.mx) | [utkukose@gmail.com](mailto:utkukose@gmail.com)

</div>

## About the course

Reinforcement learning studies how an agent learns to act from rewards [1], and the alignment of language models applies it to the most visible systems of current artificial intelligence [2, 3]. Courses usually treat the two as separate subjects and change environments at the point where the connection matters most. This course uses one environment for the whole week: TokenWorld, in which an agent generates a sequence of tokens and receives a reward when the sequence ends. A policy over TokenWorld is a language model. In its tiny form the environment can be solved exactly, so every algorithm of Days 1 to 3 is scored against a known optimum. Day 4 scales the same formalism to a neural language model of sentences and aligns it with a learned reward model, and Day 5 derives direct preference optimisation [4] and shows how optimising a proxy reward can destroy the quality it was meant to measure [5].

## Use in other courses

These materials were prepared for the value added course Reinforcement Learning and Language Model Alignment delivered at Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology. They are openly available: Anyone may use and adapt them in courses of related content, at any level, with attribution as described in the license section. Instructors who adopt them are welcome to report errors or suggest improvements through the issues of this repository.

## A look inside

Every lecture page holds an interactive scene that turns a central idea of the day into a small story with buttons. A click on a picture opens its scene on the lecture page.

<table><tr><td width="33%" valign="top"><a href="https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lecture.html#scene"><img src="days/day-01/figures/scene.png" alt="Day 1 interactive scene: The robot in the fog"></a><br><sub>Day 1: The robot in the fog</sub></td><td width="33%" valign="top"><a href="https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-02/lecture.html#scene"><img src="days/day-02/figures/scene.png" alt="Day 2 interactive scene: A mouse learns a maze"></a><br><sub>Day 2: A mouse learns a maze</sub></td><td width="33%" valign="top"><a href="https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-03/lecture.html#scene"><img src="days/day-03/figures/scene.png" alt="Day 3 interactive scene: Three ways to throw"></a><br><sub>Day 3: Three ways to throw</sub></td></tr><tr><td width="33%" valign="top"><a href="https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-04/lecture.html#scene"><img src="days/day-04/figures/scene.png" alt="Day 4 interactive scene: The leash"></a><br><sub>Day 4: The leash</sub></td><td width="33%" valign="top"><a href="https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-05/lecture.html#scene"><img src="days/day-05/figures/scene.png" alt="Day 5 interactive scene: A judge with habits"></a><br><sub>Day 5: A judge with habits</sub></td></tr></table>

The interactive labs of the first three days, as they appear in the browser.

<table><tr><td width="33%"><a href="days/day-01/README.md"><img src="days/day-01/screenshots/lab.png" alt="Day 1 interactive lab"></a><br><sub>Day 1: Foundations lab</sub></td><td width="33%"><a href="days/day-02/README.md"><img src="days/day-02/screenshots/lab.png" alt="Day 2 interactive lab"></a><br><sub>Day 2: Value-based lab</sub></td><td width="33%"><a href="days/day-03/README.md"><img src="days/day-03/screenshots/lab.png" alt="Day 3 interactive lab"></a><br><sub>Day 3: Policy gradient lab</sub></td></tr></table>

## Who the course is for

The course is written for undergraduate and graduate students of engineering and science programmes and for self-learners anywhere. It assumes no earlier programming experience: The start-here notebook and the five Python steps teach the Python needed for the course, always through the problem of the day. Basic probability and the idea of a derivative are assumed.

## Course learning outcomes

On completion, students are expected to formulate decision problems, including text generation, as Markov decision processes and solve small ones exactly, to implement and compare bandit algorithms, temporal difference methods, Q-learning, SARSA and policy-gradient methods with measured variance, stability and order of results, to report reinforcement-learning experiments over many seeds with their spread, to build a reward model from pairwise preferences and fine-tune a policy against it with a KL penalty, to derive and implement direct preference optimisation, and to diagnose reward over-optimisation and evaluate aligned models with intervals, length controls and independent judges. In Python, students are expected to write programs with lists, dictionaries, loops and functions, to compute with NumPy arrays, to plot results and to read and train small neural models written in NumPy, with their forward and backward passes.

## How the course works

Each day has an overview page in its folder, which links to four materials with distinct roles. The lecture page explains the concepts with animations, knowledge checks and review cards. The Colab notebook holds the Python step and the hands-on work, with every model of the course written in NumPy. Its exercises give immediate feedback, reference solutions sit in collapsed cells, and open questions invite observations on the results. Each section links back to the part of the lecture it applies, and each can be started on its own, because its first cell runs the code of the earlier sections. The interactive lab practises the ideas in several parts: A simulation that runs in the browser, a short classification task, a calculation by hand, and calculators that repeat the worked examples of the lecture step by step, with sliders for their numbers. The lecture page opens each part at the place where it belongs. The lab closes with a self-assessment and an exportable learning log. The PDF lecture notes collect the lecture and the Python step in one printable file. The course works in two modes. In a synchronous delivery, each day runs as one session of about three hours, following the live session plan on the overview page. In asynchronous study, the same materials are used along the study path on the overview page, which lists every activity with a suggested time.

Start with [`start-here/`](start-here/README.md) if you have never programmed, then follow the days in order. The course site at <https://utkukose.github.io/rl-alignment-veltech-NB-lecture/> links every page and keeps track of the days you have completed in your browser.

## TokenWorld, the project of the course, day by day

One project runs through the whole week. In TokenWorld an agent writes a short sentence one token at a time, and the finished sentence earns a reward for its quality: the miniature of a language model and its alignment. Each day first teaches its methods on small examples and then takes the project one stage further, in a lecture section titled *TokenWorld, stage N*. The figure shows the five stages, and the table links each stage to its section.

<p align="center"><img src="assets/tokenworld_stages.png" width="860" alt="The five stages of TokenWorld through the week"></p>

| Day | Stage of the project | Methods | Result |
|---|---|---|---|
| [Day 1](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lecture.html#tokenworld-stage-1-an-exact-optimum-as-a-yardstick) | The small TokenWorld of 4 tokens and 161 states is solved exactly. | Markov decision process, value iteration | The best sentence `the cat sat <eos>` with the return 2.75, the yardstick of the week |
| [Day 2](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-02/lecture.html#tokenworld-stage-2-values-learned-from-experience) | The rules are hidden, and values are learned from experience. | Monte Carlo, temporal differences, SARSA, Q-learning, a deep Q-network | Greedy returns 2.55 for SARSA and 2.25 for Q-learning; the deep Q-network reaches 2.75 |
| [Day 3](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-03/lecture.html#tokenworld-stage-3-a-policy-that-writes-the-best-sentence) | The probabilities of the tokens are learned directly, as a language model does. | REINFORCE, baselines, actor-critic, proximal policy optimisation (PPO) | 98 to 99 percent of the optimum |
| [Day 4](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-04/lecture.html#tokenworld-stage-4-what-the-four-stages-achieved) | A grammar of 1512 sentences and a neural language model, aligned with preferences. | Fine-tuning, a Bradley-Terry reward model, reinforcement learning from human feedback (RLHF) with a Kullback-Leibler (KL) penalty, safe and ethical alignment | Gold reward from 1.12 to 2.05; reward hacking without the penalty |
| [Day 5](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-05/lecture.html#tokenworld-stage-5-the-project-at-the-end-of-the-week) | The same model and pairs, aligned without a reward model and evaluated. | Direct preference optimisation (DPO), judges, intervals, safety rates | Gold reward 2.23 with drift from the grammar; every result stated with its judge and interval |

The final project asks each student to take a world of their own along the same path, with a gold reward that the methods never see, two alignment methods on matched data and an evaluation with intervals.

## Daily schedule

| Day | Topic | Overview | Lecture | Lab | Notebook | PDF |
|---|---|---|---|---|---|---|
| 1 | Foundations of Reinforcement Learning | [overview](days/day-01/README.md) | [lecture](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lecture.html) | [lab](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-01/lab.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkukose/rl-alignment-veltech-NB-lecture/blob/main/days/day-01/NB01_foundations_of_rl.ipynb) | [PDF](days/day-01/Day01_Lecture_Notes.pdf) |
| 2 | Value-Based Methods | [overview](days/day-02/README.md) | [lecture](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-02/lecture.html) | [lab](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-02/lab.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkukose/rl-alignment-veltech-NB-lecture/blob/main/days/day-02/NB02_value_based_methods.ipynb) | [PDF](days/day-02/Day02_Lecture_Notes.pdf) |
| 3 | Policy Gradient Methods | [overview](days/day-03/README.md) | [lecture](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-03/lecture.html) | [lab](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-03/lab.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkukose/rl-alignment-veltech-NB-lecture/blob/main/days/day-03/NB03_policy_gradient_methods.ipynb) | [PDF](days/day-03/Day03_Lecture_Notes.pdf) |
| 4 | Reinforcement Learning for Language Models | [overview](days/day-04/README.md) | [lecture](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-04/lecture.html) | [lab](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-04/lab.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkukose/rl-alignment-veltech-NB-lecture/blob/main/days/day-04/NB04_rl_for_language_models.ipynb) | [PDF](days/day-04/Day04_Lecture_Notes.pdf) |
| 5 | Direct Preference Optimization and Evaluation | [overview](days/day-05/README.md) | [lecture](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-05/lecture.html) | [lab](https://utkukose.github.io/rl-alignment-veltech-NB-lecture/days/day-05/lab.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkukose/rl-alignment-veltech-NB-lecture/blob/main/days/day-05/NB05_dpo_and_evaluation.ipynb) | [PDF](days/day-05/Day05_Lecture_Notes.pdf) |

## Python for non-programmers

Python is taught step by step alongside the content of the course, always through the tool that the problem of the day needs [6]. Every day has a Python step in its notebook, right after the setup cell, that explains one group of fundamentals with the example of the day in runnable cells and closes with a quick check and three exercises. The steps lead from values, lists, decisions and functions with a bandit on Day 1, through NumPy arrays and tables of values, vectors, the softmax and plots, to text as integers on Day 4 and objects, copies and intervals on Day 5 [7, 8].

The full sequence is described in [`PYTHON_PATH.md`](PYTHON_PATH.md).

## Assessment and submission

Assessment rests on a final project, an alignment experiment with a measurable gold reward, which consists of a coding application and a short technical report. Each day also offers an optional task, an optional research and report assignment and a self-assessment in the lab. During an active delivery of the course, components and weights are announced by the instructor at the start of the course in line with the regulations of Vel Tech University, and the daily task can be sent together with the exported learning log to utkukose@sdu.edu.tr or utkukose@gmail.com. The self-assessments are formative and do not count towards the grade.

| Component | Timing | Format | Page |
|---|---|---|---|
| Daily task and learning log | End of each day | Short coding task and exported log | each day folder |
| Final project: An alignment experiment with a measurable gold reward | One week after Day 5 | Coding application and technical report | [open](exams/final/README.md) |

A common structure for reports is given in [exams/REPORT_TEMPLATE.md](exams/REPORT_TEMPLATE.md). The syllabus in [SYLLABUS.md](SYLLABUS.md) states the policies, including the rules for using generative AI tools.

## Running the materials

The notebooks run in Google Colab through the badge of each day, with no installation: The course uses NumPy, pandas and Matplotlib, all preinstalled in Colab, and every model of the course, written in NumPy, trains on a CPU in seconds. Section 8 of Day 4 also loads a small pretrained language model, DistilGPT2, with the libraries torch and transformers, which Colab provides, and downloads it once per session, about 350 MB. For local work on Windows or Ubuntu, create a virtual environment with `python -m venv .venv`, activate it, run `pip install -r requirements.txt` and start `jupyter lab`. The lecture pages and labs open in any modern browser and keep progress only in the browser.

## Reference integrity

All references were checked before release. Entries with a link in `references/REFERENCES.md` were confirmed against that DOI or publisher page during preparation, and the remaining entries are fully citable from the details given. No locator was reconstructed from memory. As an independent check, `tools/verify_references.py` compares every entry with Crossref and OpenAlex, and the GitHub Actions workflow runs it monthly and on every change of the references.

## Citing this course

Citation metadata is provided in `CITATION.cff`. A suggested citation is:

> Kose, U. (2026). *Reinforcement Learning and Language Model Alignment: Open course materials* [Course materials]. GitHub. https://github.com/utkukose/rl-alignment-veltech-NB-lecture

## License

The course text, figures, lecture notes and interactive labs are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0), as stated in `LICENSE-CONTENT`. The code in the notebooks, labs and tools is licensed under the MIT License, as stated in `LICENSE`.

## References cited on this page

[1] Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.

[2] Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems 30 (NeurIPS 2017) (pp. 4299-4307).

[3] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. (2022). Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022) (pp. 27730-27744).

[4] Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023) (pp. 53728-53741).

[5] Gao, L., Schulman, J., & Hilton, J. (2023). Scaling laws for reward model overoptimization. In Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 10835-10866.

[6] Van Rossum, G., & Drake, F. L. (2009). Python 3 Reference Manual. CreateSpace.

[7] Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357-362. <https://doi.org/10.1038/s41586-020-2649-2>

[8] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90-95.

---

<div align="center">

**Prof. Dr. Utku Kose**

Full Professor, Department of Computer Engineering, Süleyman Demirel University, Isparta, Türkiye  
Founding Director, AI Application and Research Center (YAZEM), Süleyman Demirel University  
Head of the Computer Science Division, Department of Computer Engineering, Süleyman Demirel University  
Additional affiliations: University of North Dakota (USA), Universidad Panamericana (Mexico City, Mexico), Vel Tech University (Chennai, India)  
IEEE Senior Member, ACM Professional Member

[ORCID 0000-0002-9652-6415](https://orcid.org/0000-0002-9652-6415) | [utkukose.com](https://www.utkukose.com) | [github.com/utkukose](https://github.com/utkukose)

[utkukose@sdu.edu.tr](mailto:utkukose@sdu.edu.tr) | [utku.kose@und.edu](mailto:utku.kose@und.edu) | [ukose@up.edu.mx](mailto:ukose@up.edu.mx) | [utkukose@gmail.com](mailto:utkukose@gmail.com)

</div>

## Acknowledgments

The course draws on the work of the many researchers cited in the daily references. It was prepared for the value added course programme of the Office of International Relations of Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology, Chennai.
