# RATIONS 

## PLAN:

* Create 2800 Agents (indexes 0-2799)  -  `[DONE`
  * Give a TB Active status (Active smear posivite, Active Extra Pulmonary) - `DONE`

    \
* Allocate 1400 to CONTROL and and 1400 VITAMIN groups     `[DONE]`

  \
* Generate households with sizes from 1 to 6               `[DONE]`

  \
* Assign each household to an arm 50/50 -                  `[DONE]`
* -- Hopefully will end up slightly higher than 10,000 agents    `[DONE]`

  \
* Food Habit/BMI:
  \-- Assign BMI status
  \-- Map the BMI status to macro as part of the intervention

  \
* Assign each household a macro state based on the arm     `DONE`
  2a. set prognoses for each agent on one of the Active TB states from day 1? (maybe warm up period? do we call it warm up period? :D )
* add the bmi state to the Malnutrition class - `DONE`


## TODO:

* handle interventions differences - for indexes, control and interviention groups.
* add real data for Baseline information (bmi, macro, etc)
* scenarios

 <NEED TO ADD MORE TODOs HERE>


## Conversation with Stewart (Requests):

Do you think it would be possible to create these categories (individual properties) in TBSim?

`The categories have already been added to the simulation’s people (I am changing  some wording in the code’s comments and referring to them as Individual Properties so it matches your request).` \n  

And then we can think about the mapping between the food habit categories that we previously had from Harlem.

`I have already a 'mapping' implemented as part of an intervention;  When the intervention is added it resolves (maps) the corresponding MacroNutrientsStatus (food habits) based on the Bmi State - from there the ‘INTERVENTION’ inherits all the capabilities of the original Harlem’s NutritionChange intervention (which basically handles the food habits).`

 

One option is that they would map to each other under equilibrium -- so food habits of "standard or above" would map to BMI of "normal" or "over"; food habits of "slight below standard" would map to BMI of "mild"; and so on.

`yes indeed this could be an option - we could handle everything with the existing interventions then map them later (I need additional information here).`

 

Then if food habits changed, or in the presence of the RATIONS food basket intervention, the BMI category might change *with a delay,* so every month that you're on RATIONS you have a *slight probability* of increasing your BMI category.

`This seems to be the idea of the newly created ‘BMIchangeIntervention’  although we still need to incorporate your pointers for delay, probability of increasing your bmi etc. (currently you only specify ‘From’ and ‘To’ which state you want the intervention to be, rather than a specific probability).`

(Note that even the control group received a food basket; it was supposed to be for the TB patients, and not their household contacts, but patients probably shared it with their households, which would account for the slight increase in the control group's BMI.)

 

`yes, that has been a tricky aspect of the RATIONS study, but I understand that this affected the overall outcome of the research so we will need to figure out a way to handle it. For now, I have added a “Portion” parameter to the list of intervention’s arguments which more likely will end up impacting somehow the “delay” caused by the intervention itself, for instance if the delay on a full portion (100%)  is ‘Y’  (for the Indexes), then the household contacts should have a X% portion only with a delay of 1.2*Y times ( totally made up number)`


`and yes, in both groups the control and the intervention there will be a portion of ration applied to them, but in the case of the control group the portion was smaller (because is just the food basket meant for the index which is being shared).`