# Viva and demonstration guide

## A ninety-second explanation

Our project recognizes six human activities from ordinary webcam video. A pretrained MediaPipe model identifies 33 body landmarks. We then compute joint angles and distances after correcting the image aspect ratio and normalizing body size. We summarize two seconds of motion and train a Random Forest on these features. Our experiment compares window-mean pose features against pose plus motion statistics on participants excluded from training. We also reject incomplete tracking, confirm repeated predictions before displaying a label, and count complete squat and jumping-jack cycles. We evaluate recognition separately from repetition counts and interface stability.

## Five-minute demonstration

1. Explain the distinction between pose estimation and activity recognition. Name the pretrained detector and your feature/classification work.
2. Start the webcam and show the full body skeleton. Explain visible joint angles and the two-second window.
3. Demonstrate standing and waving, then squats. Explain complete-cycle counting and why the first partial cycle can be missed.
4. Briefly move a wrist or ankle outside the frame and show tracking rejection. Return fully into view.
5. Show the labeled dataset counts and the participant split. Open results from a real experiment and explain one confusion.
6. State one limitation and an evidence-based improvement. Use simulation only as a clearly labeled backup if camera conditions fail.

## Likely questions

**What is pose estimation?** Locating body joints in an image. It produces coordinates, not an activity label.

**Why MediaPipe?** It offers a pretrained single-person body landmark pipeline suitable for a laptop demonstration. Cite BlazePose and the current Pose Landmarker implementation. Our laptop speed still needs measurement.

**What did you train?** The activity classifier. We reused the pretrained pose detector and did not retrain or invent its neural network.

**Why not just classify individual frames?** Some actions share postures. Sitting and a squat can have similar bent-knee frames, while their motion differs. Our controlled comparison measures whether simple motion statistics help.

**Why Random Forest instead of an LSTM?** It is quick to train on compact hand-designed features and easier to explain for a small dataset. LSTMs can learn temporal patterns with more data and tuning, but are not necessary for this scope. This is an engineering choice, not proof that forests outperform deep models.

**How many features?** Eighteen geometric features per frame. The comparison model uses 18 window means. The temporal model uses four statistics per feature, giving 72 inputs.

**How do you handle body size?** Coordinates are centered at the hip midpoint and divided by torso length. Image x coordinates first receive aspect-ratio correction. This reduces translation and scale effects but does not remove all camera-view effects.

**What is your unique contribution?** A local collection and evaluation system integrating interpretable temporal pose features, quality rejection, a matched feature comparison, prediction confirmation and repetition counting. It is an application-level contribution with an empirical question. We do not claim to have invented BlazePose or Random Forest.

**Why split by participant?** Frames of the same person and neighboring windows are highly similar. Random window splitting can put near duplicates in training and testing, inflating scores. Each person appears in one split only.

**Why a validation set?** It selects the rejection threshold without consulting test labels. Test participants estimate generalization after choices are fixed.

**What does 80% vote score mean?** The forest assigns 0.8 of its aggregate class probability mass to the leading class. It is uncalibrated and should not be presented as an 80% chance of being correct.

**What is macro F1?** Calculate an F1 score for each of the six activities, then average equally. It reveals poor classes that overall accuracy can conceal.

**Why report coverage?** Rejecting difficult cases may improve accuracy on accepted cases while refusing many inputs. Coverage shows the fraction receiving an activity label. Overall accuracy still counts unknowns as errors.

**Does unknown detect all unfamiliar actions?** No. A classifier can be confidently wrong on an unseen activity. The threshold provides abstention, not a proven open-set recognition method.

**How do you count repetitions?** A finite-state machine waits for a full initial state, an active state and a return. Two thresholds reduce boundary jitter, and tracking gaps reset partial cycles.

**Can this be used for medical posture assessment?** This project measures projected geometry for a student demonstration. It has no clinical validation or reliable injury diagnosis.

**What are the main limitations?** A small local dataset, a single person and viewpoint, projected angles, occlusions, motion-window delay, and model confidence that may be miscalibrated.

**What results have you obtained?** Answer using the saved metrics from real participant recordings. Until then: implementation and software checks are complete, and participant evaluation is pending. Never substitute simulation scores.

**What would you do next?** Add more independent participants, evaluate multiple camera views, annotate transition/rep errors, run preplanned subject folds, and consider sequence models only if the dataset supports them.

## Suggested three-person ownership

Student A understands and tests the pose detector integration and geometry. Student B leads participant collection, splitting and classification evaluation. Student C owns temporal states, interface testing, report and demo coordination. Everyone should be able to explain the complete pipeline. Replace these placeholders with actual work contributions as you proceed.
