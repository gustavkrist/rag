#import "@local/jcu:0.1.0": *

#show: jcu.with(
  doc-title: "Assessment 2: LLM Project",
  author: "Gustav Kristensen",
  id: "15172944",
  course: "CP5403",
)

// #show quote: tx => { hide(tx) }
// #show figure: tx => { hide(tx) }
// #show bibliography: tx => { hide(tx) }

= Introduction

This project presents a RAG model that gives answers to questions about mechanics of the video game Terraria. The model is grounded in data from the official Terraria Wiki. The model aims to help answer questions that would otherwise require searching and browsing the wiki, consuming time that could otherwise be spend playing the game.

The model is a RAG model using the Qwen3-1.7B model @qwen_team_qwenqwen3-17b_2025 as the base model. The model is a dense 1.7 billion parameter model with reasoning capabilities. The RAG pipeline uses a vectorized database and sentence embeddings to find documents that are relevant to the user's query. The relevant documents are re-ranked by relevance with a small ranking model @noauthor_cross-encoderms-marco-minilm-l6-v2_2026, and the most relevant documents are given to the model as context.

Besides programmatic use of the model, it can also be hosted and accessed through a web page. The instructions for running the app and for reproducing the evaluation results are available in the project's GitHub repository: #link("https://github.com/gustavkrist/rag") @kristensen_gustavkristrag_2026. The GitHub repository also includes the full commit history of the development of the model.

= Target Problem

In accordance with the assessment description, I have chosen an area of application relevant to my interests. Terraria is a video game with a vast array of mechanics, items, bosses, and more. It can be very overwhelming for new players, and even veterans have a hard time remembering every detail. In public discourse, the game is often referred to as a "Wiki game", meaning you have to constantly refer to the wiki to figure out what to do (@flamyngoo_is_2025, @its_blazertron_anyone_2025, @noauthor_whats_nodate, @noauthor_other_2023). Having to tab out to the wiki often is detrimental to the immersion of the game. The problem itself it not to search out the knowledge, but especially the time spent figuring out where to get the answer. There are 6570 articles on the Terraria wiki @noauthor_terraria_2026, and without knowing exactly what information you are missing as a user, it can take some searching to find the right article, and then digest the article.

This project aims to lessen the friction caused by needing to refer to the wiki. By augmenting an LLM with the knowledge contained in the wiki, it is possible for a Terraria gamer to simply ask their question to the LLM and get an answer grounded in the articles of the wiki, with links to sources in case the user wants to read the whole article. The intended application of the RAG model developed in this project would be to deploy it in a chat box on the Terraria wiki for users of the wiki to access. The model relies on data from the wiki, so it important that the source material does not go away and keeps being developed. Making the model available on the site of the wiki ensures traffic to the wiki stays, so any add revenue they get from visits isn't taken away by providing a local model with all the data.

= Data Sources

== Data Origin and Extraction

The data for the RAG model was sourced from the Terraria wiki (#cite(label("noauthor_terraria_2026"), form: "full")). The data was fetched from the articles on the wiki using `pywikibot`, the script is available in the GitHub repository as `fetch_data.py`. `pywikibot` can extract data clean text data from MediaWiki sites @noauthor_manualpywikibot_2026, which includes the Terraria wiki. The actual script for data extraction is available in the repository as `fetch_data.py`. The script limits the extraction to articles from the following categories on the wiki:

- Boss NPCs
- Environments
- NPC NPCs
- Guides
- Liquids
- Debuffs
- Buffs
- Game mechanics
- Key items
- Boss summon items
- Permanent booster items

The choice to pick out certain categories was necessary, as the full data of the entire wiki did not produce good results. I will touch more on this later in the section about limitations. These categories were chosen because they contain the most important information needed for progression in the game and guides for new and intermediate players.

As an example, an excerpt of the extract of the article "Legacy:Guide:Skeletron Prime strategies" looks like this:

#quote(block: true)[
  Skeletron Prime, the Hardmode version of Skeletron, is intended to be the most difficult Hardmode boss, at least before 1.2. His high health, defense, damage, and multiple limbs will take a lot of punishment, and his limbs have the added effect of blocking non-piercing shots aimed at his head.


  \== Terrain preparation ==

  An ideal arena for Skeletron Prime should consist of (several) rows of wood platforms. A full stack of 99 for each row should provide ample running distance to avoid his attacks, though it never hurts to have a longer arena. The platforms should be suspended off the ground, to prevent the Prime Cannon's grenades from ever detonating. A solid ceiling should be avoided, since Skeletron Prime will often retreat a fair distance when not using his spinning attack.

  ...

  \=== Useful bug ===

  Summoning Skeletron Prime from high altitude (at least Floating Island altitude), then teleporting far enough below, can cause 1,2,3 or all of his limbs to disappear. You then only need to worry about the head and possibly one or two limbs, making the battle relatively easy. You can repeat this bug at anytime during the fight to make all of Skeletron Prime's limbs disappear.

  ...

  \== Gearing up ==


  \=== Armor ===

  Adamantite or titanium armor should be used if the player has not collected the bars necessary for Hallowed armor or Frost armor. Both the ranged and melee bonus helmets are helpful to keep on hand, as the melee bonus can be used to resist Skeletron Prime's initial attacks until his melee capacity is reduced, at which point the ranged bonus will provide extra damage without the defense loss being too much of a concern.

  ...
]

The extract is clean, usable text, but some sections are missing content because the wiki page templates cannot be fully interpolated. As the data is from a publicly editable wiki, there is not a guarantee that it is correct. The information is generally up to date and accurate, but there is no review from the game developers @noauthor_terraria_nodate.

== Data Processing

Before being indexed into the vectorized database, the data is processed into chunks. There is no data processing being done to the whole fetched articles, as they are already clean text data. The documents are chunked into chunks by extracting each section or subsection as its own data chunk. Regex is used to extract subsections into chunks with any sub-subsections inside, and main sections are chunked without including the subsections. Certain sections are not kept, as they confused the model in testing, and removing them provided better results. For all articles, the sections titled

- History
- References
- Shimmered form
- Achievements
- See also

are discarded, as well as any empty (sub)sections, such as the "Gearing up" section in the extract above. The chunks are prepended with the article title and the (sub)section title - the first section does not have a title and is given the title "Intro". As an example, one of the chunks produced from the extract above would be from the subsection "Armor":

#quote(block: true)[
  Legacy:Guide:Skeletron Prime strategies - Armor

  Adamantite or titanium armor should be used if the player has not collected the bars necessary for Hallowed armor or Frost armor. Both the ranged and melee bonus helmets are helpful to keep on hand, as the melee bonus can be used to resist Skeletron Prime's initial attacks until his melee capacity is reduced, at which point the ranged bonus will provide extra damage without the defense loss being too much of a concern.]

The article and (sub)section titles are included in the text chunk, as they may provide context that can be matched by a user query which is otherwise not present in the text chunk. The chunks are associated with a uniquely identifying ID consisting of the article ID from the wiki, combined with the sub(section) name and a number for when there are multiple (sub)sections with the same name in one article. The article's title and a URL pointing to the specific sub(section) of the article on the wiki is also saved with the chunk.

I chose to chunk the data this way, as the articles are already written with the sections and subsections being somewhat self-contained and scoped to a specific area of information. This made it a sensible choice of chunks to provide the model only the relevant information in each retrieved chunk.

== Indexing

The data is stored in a vectorized database provided by the `chromadb` python package. The default `chromadb` sentence embedder is used, which uses the `all-MiniLM-L6-v2` model from `sentence-transformers` @chroma_embedding_nodate. Each chunk is inserted in to the database, where it is embedded and then stored with the metadata mentioned before: ID, title, and URL.

== License and Use

The content on the Terraria Wiki is provided under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License @noauthor_terraria_2026. The license provides freedom to share and adapt the content, under the conditions that appropriate credit is given, the use is non-commercial, and contributions that remix, transform, or build upon the material are distributed under the same license @noauthor_deed_nodate.

The data is publicly available, so there are no concerns about private or sensitive data. For a research project like this one, it is available to use and share as part of the repository, which I have licensed under the same license. However, it is not possible to use the data for commercial use. If a model like this were to be hosted by the wiki itself however, that should be okay as they hold the licensing rights. But it would not be allowed to create a third-party commercial product out of this model.

= Methodology

// TODO:
// - [x] Elaborate on vector store and search method
//   - [x] Show example of retrieved chunks
// - [x] Discuss choice of model
// - [x] Show and talk about prompt templates
//   - [x] Both RAG and non-RAG
// - [x] Show web App screenshot

== Retrieval

As mentioned before, `chromadb` is used as the vectorized index for document retrieval. It uses the `all-MiniLM-L6-v2` model to embed chunks before storing them in the database, as well as to embed the query used to retrieve documents. I've opted to store the database persistently to avoid extra startup time when running the model. The database is generated the first time if it does not exist by processing the articles as mentioned before, and the chunks are indexed. For retrieval, the user query is used as-is, and is automatically embedded. The model fetches the 15 most similar document chunks from the index, and then keeps the ones that have a similarity distance below the set threshold. Through testing different values I found a distance of 2 to work well when combined with re-ranking.

=== Re-ranking

After retrieving up to 15 most similar documents below the distance thresholds, the documents are re-ranked using a ranking model, @noauthor_cross-encoderms-marco-minilm-l6-v2_2026. The model ranks documents by relevance to the query. Then the documents are sorted based on relevancy, with the most relevant documents first. Only the documents with a similarity score above the set threshold are kept, the rest are not used. Through testing different values, a cutoff of a similarity score of 0 works well. Next the top five most similar documents are selected, or all of them if there are fewer than five. Those up to five documents are formatted into the LLM RAG prompt as context for the model.

=== No Relevant Documents

If no relevant documents are found at all, the model generation is completely skipped and the following answer is given: "I do not know how to respond to your question.".

== Model Choice and Parameters

For this project I chose to use the model Qwen3-1.7B model. I chose this model because it was the best model I could feasibly run on my M1 MacBook Air without getting 10+ minute response times. I tried some newer models, such as in the Qwen3.5 family, which were simply too slow without CUDA hardware. I initially used Qwen3-0.6B, but I saw a significant upgrade in answer quality from using the larger Qwen-1.7B model instead, at the cost of slower answer generation.

For the model parameters, I left thinking enabled, as giving the model ability to reason seemed to give a solid improvement in the generated answers, though at the cost of much longer response times. For the sampling parameters and output length I used the parameters recommended by the Qwen team @qwen_team_qwenqwen3-17b_2025:

- Temperature: 0.6
- TopP: 0.95
- TopK: 20
- MinP: 0
- Max output length: 32,768 tokens

== Prompt Templates

=== RAG Prompt

#figure(
  box(stroke: 0.5pt, inset: 0.5em, radius: 0.5em, ```python
  rag_prompt = """\
  ## Role
  You are an assistant with expert knowledge for the video game Terraria.

  ## Instructions
  - Answer the question by using only the information in the context below.
  - If the context does not contain enough information to fully answer the question, state what you can answer and explicitely note what information is missing.
  - Cite the source for each claim by using the format [Source N].
  - Don't speculate or include information that isn't in the provided context.
  - If none of the provided context is relevant to the question, respond with: "I don't have enough information in the available documents to answer this question."
  - If the provided sources contain conflicting information, present both perspectives and cite the respective sources. Don't silently favor one source over another.

  ## Context
  {context}

  ## Question
  {user_query}"""
  ```),
  caption: [RAG prompt template],
) <rag-template>

For the RAG prompt template, I made sure to give the model a clear role and clear guidelines to follow. Importantly telling the model to only use the provided context as the basis of its reply. This helps keep the model grounded and avoid hallucinations. I also tell the model to cite its claims using the provided sources, and communicate a lack of information in the context. Making it state that information is missing instead of guessing also helps reduce hallucinations. I heavily leaned on the guidelines and examples put out by Microsoft  Azure @microsoft_develop_nodate. I chose not to give the model any instructions about the length of the generated answer, such as keep it to some number of sentences or keep it to some number of bullet points. From testing with different instructions, I found that limiting its output length caused it to fail to cite sources of claims properly, and often a degradation in the answer quality. It seems that even without an explicit instruction, it still answers briefly for some questions that only require a brief answer. The full RAG prompt template is shown in @rag-template.

=== Base Model Prompt

#figure(
  box(stroke: 0.5pt, inset: 0.5em, radius: 0.5em, ```python
  non_rag_prompt = """\
  ## Role
  You are an expert assistant for the video game Terraria.

  ## Instructions
  - Give a helpful answer to the user query.
  - Give a clear answer in 3 to 5 sentences.

  ## Question
  {user_query}"""
  ```),
  caption: [Base Model Prompt],
) <base-template>

The prompt template used for the base model is shown in @base-template. The prompt is rather simple, as the model is not provided any context to use for grounding like the RAG model. It is instructed to give a helpful answer, and asking it to keep the answer short seemed to prevent long rambles as output in my experience testing.

=== Formatting the Retrieved Documents

When the retrieved documents are put into the RAG prompt template, they are formatted in a way such that the model is able to discern between the end of one document and start of the next. They are also formatted in a way that makes it possible for the model to cite them. Each document is prepended with `[Source N]: `, with N being the number of the source starting from 1. Then the document follows, which as mentioned earlier includes the title and (sub)section title. Documents are separated by `---` with a blank line before and after, to clearly signal a break to the model. For an example of a prompt formatted with retrieved documents, see @full-context-prompt.

== Web App

#figure(
  image("images/webapp.png"),
  caption: [Screenshot of the web interface for the model with a sample question and answer],
) <web-app>

The model can be interacted with through a web interface that can be hosted locally. The web interface is made with `gradio`. In addition to the reply from the model, the answer in the web interface also includes the link to the source page and (sub)section of the wiki for each article used in the reply. A screenshot from the web app is shown in @web-app.

= Evaluation

/*
TODO:
- [x] Show example question/answering
- [x] What was done with human evaluation
- [x] What was done model-assisted
- [x] Show model judge prompt
- [x] Compare base and RAG model
- [x] Mention about the sample questions
- [x] Mention response time, retrieval latency, resource limits
*/

== Sample Questions

I manually wrote a list of sample questions, which can be found in the GitHub repository in the file `eval/sample-questions.txt`. The questions are a mix of questions relevant to the game, with some different wordings to show potential failure cases due to sensitivity in wording. Here are a few examples:

- What is a good strategy for the Solar Pillar?
- How do I increase my health capacity?
- How do I get more health?
- How do I find the mechanic?
- How do I summon Plantera?
- What are the three mechanical bosses?

To evaluate the model, I ran all the sample questions through it and saved both the model's reply, but also the context that was fed into the model. This was done for both the RAG model and the base model, so that I could also compare the two.

== Evaluation Rubric

For the evaluation, I chose to use a rubric with score of 1-5 in a few different criteria. Some of the criteria are suggested by @anderson_cp5403-prac-08ipynb_nodate. The criteria are the following:

- Correctness: Is the generated text factually correct?
- Fluency: Is the generated text fluent and cohesive?
- Perceived utility: Is the answer helpful and informative?
- Citation recall: Are the generated statements about the external world fully supported by their citations?
- Citation precision: Do the generated citations actually support their associated statements?
- Preciseness: How precisely does the generated reply answer the question?

Most of the rubric is well suited for both human and model-assisted review, though the correctness criteria requires manual review from someone knowledgeable with the source material.

=== Human Evaluation

Due to time constraints, it was not possible to do a full human review of the sample responses based on the rubric. To show how a full review may look, a subset of the sample responses were scored on correctness, since that cannot be done by a model. An expert in the domain (me) graded most of the model responses on correctness. Once again time constraints prevented a full review of correctness for all replies, as human review is very time consuming.

=== Model-Assisted Evaluation

Due to time constrains preventing a full human review, I used another LLM to score the generated replies based on the rubric, with exception of the correctness criteria. The scoring was done by the model `deepseek-v4.1-flash`. The judge prompt gives clear instructions to the judge model with the user question, context provided, generated reply for a given sample, as well as the rubric. The judge model is used through an API that supports structured output, so the output conforms to an output model that includes the scores and justification for each rubric criteria. The prompt is inspired by a judge prompt suggested by @anderson_cp5403-prac-08ipynb_nodate. The full judge prompt is shown in @judge-prompt.

#figure(
  box(stroke: 0.5pt, inset: 0.5em, radius: 0.5em, ```python
  JUDGE_PROMPT = """\
  You are evaluating a question-answering system.

  Question: {query}

  Context provided to the system:
  {context}

  System's answer: {reply}

  Score the answer on the following criteria on a scale of 1-5:
  Fluency: Is the generated text fluent and cohesive?
  Perceived utility: Is the answer helpful and informative?
  Citation recall: Proportion of generated statements about the external world that are fully supported by their citations
  Citation precision: Proportion of generated citations that support their associated statements
  Preciseness: How precisely does the generated reply answer the question?

  Reply with the score and one sentence of justification for each"""
  ```),
  caption: [Judge model prompt],
) <judge-prompt>

== RAG Model Performance

The RAG model performs somewhat mediocre when looking at it by itself. Most of the scores are only a bit higher than 3, which is in the middle given the 1-5 rating range. The full scores and summary statistics are shown in @eval-stats. The fluency is very high, and the model is not always correct given the mean correctness score of 3.62.

#figure(
  include "tables/eval-stats.typ",
  caption: "RAG Model Evaluation Results",
) <eval-stats>

== Base and RAG Model Comparison

The outlook is better when compared to the base model that doesn't use RAG. The full scores of the base model is shown in @eval-stats-no-rag, and the mean scores are shown side-by-side in @eval-comparison. As can be seen in @eval-comparison, the RAG model performs better in every marking criteria, and quite substantially so in every criteria except fluency. The base model is almost always entirely factually wrong, which makes it useless for the target problem. This shows that the RAG model does provide a substantial improvement in answer quality and accuracy over the base model. A sample of the generated answer for each model is can be seen in @sample-reponse. The base model typically answers more briefly, in a single paragraph and very confidently. However, it is almost entirely factually wrong in every observed case.

#figure(
  include "tables/eval-stats-no-rag.typ",
  caption: "Base Model Evaluation Results",
) <eval-stats-no-rag>

#figure(
  include "tables/eval-comparison.typ",
  caption: "Comparison of mean scores for RAG and Base model",
) <eval-comparison>

== Comparison to Frontier Model

I ran a few sample questions through Opus 5.5 hosted through #link("https://copilot.cloud.microsoft"). The responses can be seen in @opus-resonses. For questions that refer to elements that are specifically from Terraria and the model can be sure enough of it, it provides a good response. However, for ambiguous questions, it will ask for confirmation or follow-up questions. This creates friction for a user that just wants a direct answer. The benefit of the RAG model is that it already knows what the domain is that the user is asking about. Additionally, for some questions that contain no words or elements specific to Terraria or games in general, Opus 5.5 responds with a completely unrelated answer, while the RAG model gives an answer relevant to Terraria. This can for example be seen with the question "Can I swim in lava?". In this case the RAG model gives a relevant answer about how you can possibly do it in the game, while Opus 5.5 responds with advice about how you cannot swim in lava in real life.

== Limitations of the Model

The RAG model is very sensitive to specific wording. For example, the query "How do I get more health?" cannot be answered, while "How do I increase my max health?" gives and okay response, and "How do I increase my health capacity?" gives a good response. The questions are all asking the same thing, but to get a good response from the model, the wording has to be very precise. Another example of this is the query "How do I become immune to confusion", which the model cannot answer, while the query "How do I become immune to being confused?" does give a generated response. The limitation lies in the document retrieval process, as the queries that the model cannot answer do not match any documents.
// TODO: ^ this should be in the discussion as query rewriting could potentially help alleviate this problem

Despite being told to always cite sources for claims, the model sometimes fails to do so properly. Sometimes it puts all the source attributions together at the end of the message, making it impossible to discern which claim is supported by which source:

#quote(block: true)[
  If you die in Hardcore mode in Terraria, the following happens:
  1. \*\*Character File\*\*: Your character's files are placed into the recycle bin (on Xbox) and can be recovered if you log back in. On other platforms, characters are permanently deleted.
  2. \*\*Items\*\*: All items you were carrying (including coins, ammo, and portable storage items) are dropped at the location of your death. Items in chests remain in the world, while portable items are deleted.
  3. \*\*Ghost State\*\*: You transform into a ghost, which allows you to fly, pass through walls, and chat with other players. However, you cannot interact with the environment (e.g., open doors unless Smart Doors are enabled).
  4. \*\*Respawn\*\*: Hardcore characters respawn like Mediumcore characters, dropping items upon death. If you die and then leave the world, your dropped items may despawn.
  5. \*\*Special Conditions\*\*: With Celebration Mk 10 and The Constant seeds active, Hardcore characters have reduced maximum health upon death. They only die permanently if their health reaches zero or below; otherwise, they respawn. The Extra Life achievement is possible with this setup.

  \*\*Note\*\*: This applies to Hardcore mode, which is distinct from Hardmode (a progression term for post-Wall of Flesh worlds). The ghost state and item behavior are specific to Hardcore.

  \*\*Source\*\*: [Source 1] [Source 2] [Source 3] [Source 4]
]

On rare occasion, the model will completely omit source attributions, such as this case:

#quote(block: true)[
  To summon Plantera in Terraria, you must destroy a \*\*Plantera's Bulb\*\* after all three mechanical bosses (Golem, Lihzahrd, and Plantera) have been defeated. Here's the process:

  1. \*\*Post-Boss Defeat\*\*: Once the three mechanical bosses are defeated, Plantera's Bulbs spawn in the Underground Jungle. These bulbs emit a faint pink light and can be mined, exploded, hit by a Clentaminator, or physically hammered to trigger their summoning.
  2. \*\*Destroy the Bulb\*\*: Breaking the bulb (using a pickaxe, drill, or explosives) summons Plantera. The bulb requires 2 contiguous flat jungle grass tiles below it to spawn and can break other jungle plants (e.g., Life Fruit) to make space.
  3. \*\*First Bulb\*\*: An unmodified jungle will automatically spawn one Plantera's Bulb upon defeating the final mechanical boss. This bulb always spawns on the side opposite the Dungeon point, near the natural jungle.

  \#\#\# Notes:
  - Plantera's Bulbs cannot spawn if other Plantera's Bulbs intersect a 301×301 square centered on the left tile of the bulb.
  - The bulb's summoning is probabilistic (3/5000 chance per update) and requires careful placement to avoid unintended boss fights.

  This method is the primary way to summon Plantera, as described in the context.
]

The complete lack of citing was only seen once in the sample answers. These issues with citing could be due potentially be fixed by further tuning the prompt.

== Efficiency

The experiments have been run on my M1 Macbook Air, which is becoming dated at this point. As a result, the chosen language model is the largest one it can run reasonably, but it is rather slow. For queries that only matched one or two relevant documents, or if the retrieved documents are very short, the answer time is around 30-40 seconds. For very queries that return the maximum possible of five documents, and if the documents are on the longer side, response times can get up to a couple minutes. Running the model on better hardware, such as a dedicated GPU would make a lot of difference in this case. The actual retrieval only takes a second or two, the delay primarily comes from the model. On one hand, it does lower usability a fair bit with such high answer times, on the other hand, the model can be hosted on very light hardware.

= Discussion

/*
TODO:
- [ ] What worked
- [ ] Limitations
*/

== Limitations

While the RAG model does provide a substantial improvement in the target domain over the base model, it is not without issues. Some limitations of the model's capabilities were mentioned in the previous section. In additions of the model, certain limitations of the project as a whole also have an impact on the model performance.

The RAG model works only for a subset of the knowledge contained in the Terraria wiki. As mentioned earlier in the report, only articles from certain categories were chosen as data for the model. I did initially use every article from the entire wiki, but there are a lot of articles with only a sentence or two, and they would often get retrieved while not relevant to the query. Narrowing down the articles resulted in much better model responses, but also a less broad knowledge base for the model. In addition, there is some information on the wiki pages that does not get extracted by the `pywikibot` library, and those sections end up empty in the output. This information is also lacking, meaning there are many questions, for example about how to craft items, which the model cannot answer.

As mentioned, time constrains meant it was not possible to do a full human evaluation of the models. Ideally sample questions, context, and replies would be evaluated by people from the target audience, Terraria players. Since they are the target audience for the model, their evaluation would be most relevant for improving the model's responses. Ideally there should be more time for more iteration loops of evaluation followed by changes to the model based on the evaluation. That would result in a model more tailored towards the target audience.

The response time of the model is quite slow as mentioned, but this is in large part due to running the experiments on an M1 Macbook Air. If the model was run on more powerful hardware, the response time should go down significantly. It would even be possible to run a larger model with even a consumer GPU, which could result in both better response time and better responses.

== Future Work

To build on this project, it would be interesting to evaluate the model further against current frontier cloud models. The RAG model produced from the project has the benefit that it can be run locally and would be very cheap to host without requiring much enterprise hardware. On the other hand, cloud models may include the data from the Terraria wiki in their training data these days and are often able to answer questions about Terraria directly. However, as shown, they often ask follow-up questions for additional information or confirmation. The RAG model always provides an answer in one shot. It would be good to do a more thorough evaluation of the cloud model's ability to give a one-shot answer, as well as the other rubric metrics the RAG model was judged on.

The RAG model itself is only the baseline for allowing gamers to interact with the knowledge contained in the Terraria wiki more fluently. A useful extension of this project would be a speech-to-text and text-to-speech pipeline surrounding the RAG model, which would entirely prevent the user from having to leave the game. The user would be able to fluently ask questions and get answers grounded in the knowledge source, without having to leave the game at all.

= Conclusion

The RAG model answers questions about Terraria decently, scoring an above middle score on average in every metric of the evaluation rubric. It offers a big improvement over the base model, which practically never answers accurately. The model provides a new, more frictionless way to interact with the knowledge in the Terraria wiki. A Terraria player who would previously be tabbing out to search around the wiki can simply ask their question to the model and get a reply. There is still room for improvement, both in terms of accuracy of the answers, response time, and knowledge of the model. But this serves as a solid baseline.

Despite current frontier models being able to answer questions about Terraria, this RAG model provides certain benefits: it is easy to run on local hardware, answers questions in one shot without follow-up questions, and it gives domain specific answer to vague questions that don't mention anything within the domain.

Further extensions to the model could include a speech-to-text and text-to-speech pipeline, such that a Terraria player would be able to interact directly with the knowledge base of the Terraria wiki by speaking to the model. This would provide an even more frictionless experience, where they would not even have to leave the game and stop playing to type to get their questions answered.

#bibliography(
  "bibliography.bib",
  style: "american-psychological-association",
  full: false,
  title: "References",
)

#heading(numbering: none)[Acknowledgement of AI Use]

No content generated by AI technologies has been used in this assessment.

#show: appendix

= Example Prompt With Full Context <full-context-prompt>

Given the user query "How do I summon Plantera?", after retrieving the five most relevant documents in the manner described earlier and formatting them into the prompt, the model is given the following context:

#quote(block: true)[
  ```
  ## Role
  You are an assistant with expert knowledge for the video game Terraria.

  ## Instructions
  - Answer the question by using only the information in the context below.
  - If the context does not contain enough information to fully answer the question, state what you can answer and explicitely note what information is missing.
  - Cite the source for each claim by using the format [Source N].
  - Don't speculate or include information that isn't in the provided context.
  - If none of the provided context is relevant to the question, respond with: ""I don't have enough information in the available documents to answer this question.""
  - If the provided sources contain conflicting information, present both perspectives and cite the respective sources. Don't silently favor one source over another.

  ## Context
  [Source 1]: Guide:Crafting a Zenith - Seedler
  The Seedler is a Hardmode sword that you can obtain from defeating Plantera, with a 1/8 (12.5%) chance of dropping. When used, it fires Seedler Nut projectiles that explode into Seedler Thorns upon a period of time, or when hitting an enemy.
  Plantera can only be summoned by breaking Plantera's Bulbs, a naturally spawning object that can generate after all mechanical bosses are defeated. See Guide:Plantera strategies for more information on how to defeat Plantera.

  ---

  [Source 2]: Plantera - Summoning
  After all three mechanical bosses have been defeated, Plantera's Bulbs will randomly spawn in the Underground Jungle (indicated by the status message ""The jungle grows restless...""). These bulbs emit a faint pink light in dark areas and immediately summon Plantera nearby upon being destroyed, usually a fair distance off-screen.

  ---

  [Source 3]: Golem - Summoning
  Golem is summoned by using a Lihzahrd Power Cell at the Lihzahrd Altar located in the final chamber of the Jungle Temple, at any time, and only if the current world is in Hardmode after Plantera is defeated. Upon pressing ⚷ Open / Activate on the Lihzahrd Altar, one Lihzahrd Power Cell is consumed from the player's inventory and Golem is summoned directly above the altar.
  The Lihzahrd Altar can be relocated after being mined with a Picksaw or better. However, Golem will become enraged while outside of the Underground Jungle.

  ---

  [Source 4]: Plants - Plantera's Bulb
  Plantera's Bulb is a pink plant in the Underground Jungle that will begin to grow after all 3 mechanical bosses have been defeated (1 will suffice on the Nintendo 3DS version) and the world is in Hardmode. It will summon Plantera when mined with a pickaxe/drill, broken by explosives, hit by a Clentaminator with Solution, or if the Jungle grass under it is hammered or converted to another block. Though it cannot be broken by other weapons or projectiles, caution should still be taken if these conditions are met to prevent any unwanted boss fights. Plantera's Bulbs require 2 contiguous flat jungle grass tiles below the surface line to spawn on, and can break other jungle plants besides Life Fruit and Plantera's Bulbs to make room for their spawn. They cannot spawn if there are any other Plantera's Bulbs intersecting a 301×301 square centered on the left tile the bulb attempts to spawn on, and if conditions are met, they will spawn with a 3/5000 (0.06%) chance per update.
  An unmodified jungle will generally have Plantera's Bulbs spawn on average every 6 to 20 minutes, depending on world size. Due to requiring flat ground to spawn on, Plantera's Bulbs can spawn much more quickly if the player has already mined long horizontal tunnels in the Underground Jungle, or if they make a farm for them.
  One Plantera's Bulb will automatically spawn once the final mechanical boss is killed for the first time. This first bulb will always spawn on the side opposite the Dungeon point (thus, near the natural jungle) between 15% and 35% the total width of the world, or 65% and 85% the total width of the world depending on side.

  ---

  [Source 5]: Guide:Walkthrough/Hardmode - Plantera
  While observing the jungle at this stage, the player may find a mysterious pink bulb. Destroying this bulb with a drill or pickaxe summons the boss Plantera.

  Upon defeating Plantera, the message ""Screams are echoing from the dungeon..."" appears, and powerful new enemies are added to the Dungeon. The boss also drops the Temple Key, which is used to open the Jungle Temple.
  Additionally, the Truffle NPC starts selling the Autohammer that is used to craft a ranged Shroomite armor via Glowing Mushrooms and Chlorophyte bars.

  ## Question
  How do I summon Plantera?
  ```
]

= Sample RAG and Base Model Response <sample-reponse>

Given the user query: "What is a good strategy for the Solar Pillar?", the RAG model responded with:

#quote(block: true)[
  A good strategy for the Solar Pillar in Terraria involves several key tactics, as outlined in the provided context:

  1. \*\*Stay Grounded\*\*: Avoid airborne players, as Crawltipede minions target them heavily. Use a skybridge to shoot down enemies on the ground and the pillar without being targeted by Crawltipedes [Source 3].
  2. \*\*Use Water for Mobility\*\*: If equipped with a Neptune's Shell, Gills Potions, or a large pool of water under/next to the pillar, you can move off the ground without being targeted by Crawltipedes. This also slows enemies, making them easier to target [Source 3].
  3. \*\*Build Defensive Structures\*\*: Construct a small building with doors and platforms, two floors tall, near the pillar’s spawn area. This blocks enemies and forces them into a confined space, simplifying defense [Source 3].
  4. \*\*Leverage Ranged Weapons\*\*: Use a ranged weapon to lock in Corites and target their tail. Homing weapons like the Vortex Beater with Chlorophyte Bullets or Razorblade Typhoon can efficiently attack Crawltipedes [Source 4].
  5. \*\*Fly Diagonally for Speed\*\*: Equip wings (e.g., Frog Leg or Soaring Insignia) and fly diagonally to outrun Crawltipedes. Use homing weapons to attack their tails while airborne [Source 3].
  6. \*\*Prioritize Pillar Order\*\*: Fight the Stardust Pillar first to obtain the Stardust Dragon Staff, which is essential for defeating remaining pillars. The Solar Pillar requires a different strategy than others, with the Vortex and Nebula pillars being easier to handle [Source 2].

  \*\*Order of Fighting\*\*: The Stardust Pillar is recommended as the first to defeat for the Stardust Dragon Staff, followed by the Solar Pillar, and finally the Vortex or Nebula pillars if needed [Source 5].

  \*\*Additional Tips\*\*:
  - The Selenian can reflect projectiles, and the Solar Pillar’s enemies (e.g., Corites and Crawltipedes) are hindered by blocks, making the pillar easier to control with proper building [Source 4].
  - The Solar Pillar drops Solar Fragments, which can be used for crafting, but the focus is on defeating enemies rather than collecting fragments [Source 4].

  These strategies emphasize defensive positioning, mobility, and weapon efficiency to maximize survival and victory.
]

Given the same query, the base model responded with:

#quote(block: true)[
  A good strategy for the Solar Pillar involves using a weapon like the \*\*Solar Sword\*\* or \*\*Lava Sword\*\* to pierce through its protective shell. Focus on attacking when the pillar is vulnerable, such as when it swings its arms or moves toward you. Avoid direct confrontation with the lava surrounding it; instead, use a shield to deflect heat and protect your character. Position yourself near the pillar’s base and attack while it’s stationary, as it can be weakened by concentrated strikes. Always prioritize timing and environmental awareness to maximize damage and minimize risk.
]

Given the query "Can I swim in lava?", the RAG model responded with:

#quote(block: true)[
  Yes, you can swim in lava in Terraria, but it requires specific items or a mount. The context states that \*\*Gills Potions are not needed\*\* for swimming in lava, and that Obsidian Skin Potions, Flipper Potions, or a Flipper can aid movement in lava [Source 1]. Additionally, the \*\*Superheated Blood\*\* mount allows riding a Lava Shark Mount, which enables swimming in lava without taking damage, though the mount itself is obtained via Obsidian Crates [Source 2].

  However, the context does not explicitly confirm whether the player can swim in lava \*\*without\*\* these items or a mount. The ability to swim in lava is implied through the mount's functionality, but the necessity of items like Obsidian Skin or Flipper Potions is noted as a workaround.

  \*\*Answer:\*\*
  Yes, you can swim in lava in Terraria using Obsidian Skin Potions, Flipper Potions, or a Flipper, or by riding a Lava Shark Mount. However, the context does not explicitly confirm whether swimming in lava is possible without these items or a mount. [Source 1][Source 2]
]

Given the same query, the base model responded with:

#quote(block: true)[
  In Terraria, you cannot swim in lava because it is a dangerous, hot environment that can instantly damage or kill you. The game's design prevents players from interacting with lava, as it is a hostile element that poses significant risks. Instead, players are encouraged to explore safe areas and avoid lava sources to survive.
]

= Sample Opus 5.5 Responses <opus-resonses>

Given the question "What is a good strategy for the Solar Pillar?", Opus 5.5 through #link("https://copilot.cloud.microsoft") responded with:

#quote(block: true)[
  Assuming you mean the Solar Pillar in Terraria, here's a strategy that works well.

  \# Preparation
  - Arena: Build a long, flat platform or a few layers of platforms near the pillar. That gives you room to dodge and lets you see the Crawltipedes coming.
  - Mobility: Use wings or a hoverboard and a good dash, like the Master Ninja Gear. Staying airborne and mobile is the main way you survive.
  - Buffs: Ironskin, Endurance, Regeneration, Lifeforce, Wrath or Rage, and a Heart Lantern plus a campfire if you're fighting near your arena.
  - Healing: Greater or Super Healing Potions, and a Nurse nearby if you can manage it.
  \# Enemy priorities
  - Crawltipede: This is the biggest threat because it hits very hard and keeps coming. You can't kill it permanently, so dodge it. It struggles to reach you while you're grounded on a solid surface. Stay aware of where it is at all times.
  - Selenians: They spin and reflect projectiles. Hit them when they stop spinning, or use melee or homing weapons.
  - Drakomires and their Riders: Kill the rider first, then the Drakomire.
  - Corites: They dash at you. Take them out quickly or dodge them.
  - Sroller and Drakanian: These are low priority. Clear them as they come.
  \# Weapons and class tips
  - Ranged: Chlorophyte Bullets or the Tsunami work well. Homing projectiles avoid trouble with Selenians.
  - Magic: Razorblade Typhoon and Bubble Gun are strong. Watch your mana.
  - Summoner: Stardust Cell or Xeno Staff minions, plus a Desert Tiger, let you focus on dodging.
  - Melee: Terra Blade, Influx Waver, or yoyos like the Terrarian are good. The Solar Pillar is often the hardest pillar for melee, so bring defense.
  \# Strategy
  - Fight near the pillar so enemies spawn in front of you.
  - Keep moving horizontally. Standing still invites Crawltipede hits.
  - Once the shield is down after 100 kills (150 in Expert/Master), the pillar itself is easy. Burst it down while dodging its fireball rain.
  - Many players do Solar last because it's the toughest pillar, so you'll have better gear from the other pillars first. For melee classes, it's still the target because its drops lead to Solar Flare armor.

  If you meant a different game, tell me which one and I'll adjust.
]

Given the question "What are the modifiers of expert mode?", Opus 5.5 through #link("https://copilot.cloud.microsoft") responded with:

#quote(block: true)[
  Could you tell me which product or game you mean by "expert mode"? The term is used in many places, for example:

  - Terraria: Expert Mode changes enemy stats, boss AI and drops.
  - Software or tools: Some apps have an "expert mode" with advanced settings, such as camera apps, installers and trading platforms.
  - Other games: Many titles have an Expert difficulty with their own modifiers.

  Once I know which one, I can list its modifiers. If it's from an internal tool or document at your organization, the name of that tool or document would help too.
]

Given the question "Can I swim in lava?", Opus 5.5 through #link("https://copilot.cloud.microsoft") responded with:

#quote(block: true)[
  No, swimming in lava would kill you, and you wouldn't actually be able to swim in it.

  - It's extremely hot. Lava is usually about 700–1,200 °C (1,300–2,200 °F). The radiant heat alone can cause severe burns and set clothing on fire before you even touch it.
  - You wouldn't sink. Lava is about three times denser than water and very thick, so a body would rest on the surface instead of going under. That's very different from the movie scenes where people sink.
  - It would be fatal within moments. Touching it would cause catastrophic burns right away. Volcanic gases near lava, such as sulfur dioxide, are also toxic.

  If you want to see lava up close, guided tours at places like Hawaiʻi Volcanoes National Park or Iceland's eruption sites let you watch from a safe distance.
]
