# Syllabus: Reinforcement Learning and Language Model Alignment

## Course information

| Item | Detail |
|---|---|
| Course | Reinforcement Learning and Language Model Alignment |
| Code | VTR UGE 21, value added course |
| Institution | Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology, Chennai, India |
| Credits | L-T-P-C 1-0-0-1 |
| Format | Five days, one module per day, synchronous sessions with materials for asynchronous study |
| Instructor | Prof. Dr. Utku Kose |
| Language | English |

## Course description

Reinforcement learning studies how an agent learns to act from rewards , and the alignment of language models applies it to the most visible systems of current artificial intelligence . Courses usually treat the two as separate subjects and change environments at the point where the connection matters most. This course uses one environment for the whole week: TokenWorld, in which an agent generates a sequence of tokens and receives a reward when the sequence ends. A policy over TokenWorld is a language model. In its tiny form the environment can be solved exactly, so every algorithm of Days 1 to 3 is scored against a known optimum. Day 4 scales the same formalism to a neural language model of sentences and aligns it with a learned reward model, and Day 5 derives direct preference optimisation and shows how optimising a proxy reward can destroy the quality it was meant to measure .

## Prerequisites

No programming experience is required. Basic probability, the idea of an expected value and the idea of a derivative are assumed. Students without programming experience should complete the start-here notebook before Day 1.

## Learning outcomes

On completion, students are expected to formulate decision problems, including text generation, as Markov decision processes and solve small ones exactly, to implement and compare bandit algorithms, temporal difference methods, Q-learning, SARSA and policy-gradient methods with measured variance, stability and order of results, to report reinforcement-learning experiments over many seeds with their spread, to build a reward model from pairwise preferences and fine-tune a policy against it with a KL penalty, to derive and implement direct preference optimisation, and to diagnose reward over-optimisation and evaluate aligned models with intervals, length controls and independent judges. In Python, students are expected to write programs with lists, dictionaries, loops and functions, to compute with NumPy arrays, to plot results and to read and train small neural models written in NumPy, with their forward and backward passes.

## Daily plan

| Day | Module | Python step |
|---|---|---|
| 1 | Foundations of Reinforcement Learning | Values, variables, lists and decisions |
| 2 | Value-Based Methods | NumPy arrays and tables of values |
| 3 | Policy Gradient Methods | Vectors, the softmax and learning curves |
| 4 | Reinforcement Learning for Language Models | Text as integers |
| 5 | Direct Preference Optimization and Evaluation | Objects, copies and the DPO loss |

## Learning activities and workload

Each day combines about three hours of synchronous session with about three to five hours of individual work on the notebook, the lab, the challenges and the daily task. In asynchronous study, the study path of each day gives a suggested time for every activity, about six to eight hours per day in total. The final project needs about twenty hours.

## Assessment

Assessment rests on a final project, an alignment experiment with a measurable gold reward, which consists of a coding application and a short technical report. Each day also offers an optional task, an optional research and report assignment and a self-assessment in the lab. During an active delivery of the course, components and weights are announced by the instructor at the start of the course in line with the regulations of Vel Tech University, and the daily task can be sent together with the exported learning log to utkukose@sdu.edu.tr or utkukose@gmail.com. The self-assessments are formative and do not count towards the grade.

## Policy on generative AI tools

Generative AI tools may be used for explanation, debugging and drafting, under three conditions. Every use is declared in the statement on tools of the report, every output that enters the submitted work is checked by the student, and every reference is verified against its source. A fabricated reference, a fabricated result or undeclared generated text is treated as a breach of academic integrity.

## Academic integrity

Work submitted for assessment must be the student's own. Collaboration in class is encouraged, while code and reports for the final project are written individually or by the declared pair. Sources are cited for every idea, figure, dataset and piece of code taken from others.

## Accessibility

All pages run in a browser without installation and work on phones. Labs can be used with the keyboard, and animations respect the reduced-motion setting of the operating system. Lecture notes are also provided as PDF.

## Main textbooks and open resources

The course is self-contained, and every source is cited where it is used. The standard textbook is the second edition of Sutton and Barto, freely available from the authors at <http://incompleteideas.net/book/the-book-2nd.html>.
