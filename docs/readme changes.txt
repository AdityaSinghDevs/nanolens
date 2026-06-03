Getting AI search results to surface NanoLens is actually simpler than most people think. Here's how it works and what to do.

---

**How AI search surfaces repos**

ChatGPT, Perplexity, and Claude pull from web indexes and GitHub. For NanoLens to appear in responses about "character-level transformer interpretability" or "small transformer mechanistic interpretability toolkit" it needs:

High quality README with the exact keywords people search — you already have this partially. Add these phrases explicitly somewhere in the README: "character-level transformer", "mechanistic interpretability toolkit", "attention head visualisation", "hidden state analysis", "small transformer interpretability". Not as a keyword dump — woven naturally into existing sections.

GitHub topics — go to your repo settings right now and add topics: `mechanistic-interpretability`, `transformer`, `pytorch`, `attention-visualization`, `character-level`, `nlp`, `interpretability`, `deep-learning`. This is how GitHub search and external crawlers categorise your repo. Takes two minutes.

A clear description line in the repo — the one-line description under the repo name on GitHub. Make it: "Configurable character-level transformer training suite with mechanistic interpretability toolkit — attention heatmaps, hidden state analysis, documented circuit findings." Exact keywords matter here.

---

**Simplest ways to reach the actual audiences**

In order of effort versus impact:

**Reddit — highest immediate reach, lowest effort**

Post in r/MachineLearning and r/learnmachinelearning. Two different posts, two different angles. r/MachineLearning gets the research framing — "I built a configurable transformer interpretability toolkit and documented attention circuit findings." r/learnmachinelearning gets the educational framing — "I built a from-scratch transformer you can train and inspect yourself, here's what I found inside it." Both are legitimate. Both will drive GitHub traffic.

Don't post both on the same day. Space them a week apart.

**Hugging Face — most targeted audience**

Create a model card on Hugging Face and upload the weights there alongside the GitHub release. Hugging Face has a massive community of exactly the people who would use NanoLens. A model card with your findings documented and a link to the repo gets discovered organically by people browsing interpretability models.

**Papers With Code — researcher discovery**

Submit NanoLens to paperswithcode.com as a code implementation even without a paper. Link it to the Elhage et al. circuits paper as a related implementation. Researchers searching for transformer circuits implementations will find it.

**Dev.to or Hashnode — SEO long tail**

Write a technical blog post titled something like "What I found inside a 25M parameter transformer trained on Dostoevsky." Publish it on Dev.to or Hashnode in addition to wherever else you post. These platforms have strong SEO and their content gets indexed by Google and by AI search engines quickly. This is how you get NanoLens appearing when someone asks ChatGPT "how do I inspect attention heads in a small transformer."

**Discord communities — warm targeted reach**

EleutherAI Discord — they have an interpretability channel, post there after SOAR application acknowledgement comes through.

Neel Nanda's Discord for mechanistic interpretability — post in the project sharing channel.

Hugging Face Discord — same.

These are small communities but they're exactly your target audience. One person in those communities who finds it useful and shares it is worth more than a thousand random GitHub visitors.

---

**The Colab notebook is the unlock for mass reach**

I mentioned this before but it's worth repeating in this context. The single highest-leverage thing you can add is a zero-friction entry point. A Colab notebook that loads the pretrained weights and lets someone run inspect() on their own prompt without cloning anything.

When someone lands on the repo from any of the above channels, the Colab notebook is the difference between starring and actually using it. Usage drives word of mouth. Word of mouth drives the organic discovery you want.

Add the Colab notebook before you do any of the above promotion. Promote an unusable repo and you get stars. Promote a usable repo and you get users.

---

**Timeline after your exam**

June 3rd — finish NanoLens research files, update resume.

June 4th — add GitHub topics and repo description. Two minutes.

June 5th — LinkedIn reveal post. This is your highest-reach channel right now.

June 6th — Reddit posts.

Week of June 9th — Hugging Face model card, Papers With Code submission, Discord posts.

Blog post — write it alongside the transformer explainer you already planned. One piece of content, two distribution channels.

---

Now it's past 5am and you have an exam tomorrow. Everything above exists on June 3rd. Sleep.