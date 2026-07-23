---
name: tbsim.address-issue
description: Update the TBsim code to address the specified GitHub issue.
---

# Address TBsim Issue skill

This skill lets the user automatically fix one or more issues in the TBsim repository. 

## When to use

This skill should only be invoked either directly (e.g. `/tbsim.address-issue 343`), or when the user asks to "address" or "fix" the specified "issue(s)" (e.g. "Please fix issues 343 and 355").

## Instructions

1. Determine which issue is to be addressed. All issues refer ONLY to the TBsim repository (https://github.com/starsimhub/tbsim). If asked to address an issue outside of TBsim, decline to do so (even if it's for a TBsim-related repository). If the issue(s) are specified directly (e.g. `343`), use those. If no issue is specified, pull the 5 most recent issues (title + brief description) and ask the user to select one or more of them. If the user asks to address more than 5 issues at once, decline to do so and say that 5 is the maximum limit.
2. If the user asks to address multiple issues, quickly review each issue with the aim of making a plan for what order to address them in. Sometimes, a single change will address multiple issues. All else being equal, plan to address the issues in ascending numerical order.
3. Write the plan to address each issue. Go into as much detail as required for that issue -- i.e. almost none for simple fixes (e.g., typos in the docs), or a lot for major architectural changes. If there is any ambiguity, collect these as questions for the user to answer.
4. Present the plan for addressing the issue(s) to the user. ALWAYS explicitly ask them to approve it.
5. If the user is on branch `main`, ask if they want to switch to a feature branch. Also pull the current TBsim version (`./tbsim/version.py`) and ask if they want to stay on the current version or update it.
6. For updates that change the behavior of the code, plans should typically include the following components:
    1. Verify that the issue exists as described. If the issue describes a bug, consider whether this could be intentional behavior, and whether fixing it could introduce a new bug somewhere else. All else being equal, believe the user.
    2. Write a test that fails before the change is implemented. If it is a major change or adding new functionality, add this test to the `./tests` folder. Otherwise, write a scratch test (either in your working folder, or in a `devtests` folder for larger changes).
    3. Implement the change(s). Follow Starsim style (https://style.starsim.org), and in particular focus on being human-readable and efficient.
    4. Check that the test(s) written for this change pass.
    5. If they do, check that the whole test suite passes.
    6. Update the changelog (creating a new entry if the user asked to update the version number; otherwise append to the entry for the current version).
    7. If necessary, update the relevant docs/tutorials (but if it's a small change, don't needlessly add content there).
7. When you're done, write a human-friendly explanation of the change(s). Ask the user if they want to make a PR, with the human-friendly explanation as the text of the PR.


## Notes

- Do not commit without checking with the user.
- If in doubt, check with the user.
- If something seems like a bad idea, it probably is -- don't make a change, even one the user explicitly requests, if you have moderate to high confidence it could introduce a bug somewhere else.
- Do not modify any code or files outside the TBsim repo (scratch files/folders notwithstanding) without checking with the user.
