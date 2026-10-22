# Evaluation report

| Metric | Value |
|---|---|
| Questions | 20 |
| Refusal accuracy (answered when answerable, refused when not) | 100% |
| Answerable questions answered | 100% |
| Unanswerable questions refused | 100% |
| Correct document + page cited | 100% |
| Faithfulness (LLM judge, 0-1) | 1.00 |
| Correctness vs reference (LLM judge, 0-1) | 1.00 |
| Unanswerable questions answered anyway | 0 |
| p95 latency | 2.66 s |
| Errors (service failures) | 0 |

Judge scores average over the 15 answers the bot produced (0 judge failures excluded); refused questions are not scored. The judge is an LLM, typically the same model as the generator — treat its scores as indicative and spot-check the per-question table.

## Per question

| ID | Question | Behaviour | Source | Faithful | Correct | Cited |
|---|---|---|---|---|---|---|
| q01 | How do I factory reset the S7? | ✅ | ✅ | 1.00 | 1.00 | product-guide.pdf p.4 |
| q02 | How long does the battery last in Eco mode? | ✅ | ✅ | 1.00 | 1.00 | product-guide.pdf p.1 |
| q03 | Can I connect the robot to a 5 GHz Wi-Fi network? | ✅ | ✅ | 1.00 | 1.00 | product-guide.pdf p.2 |
| q04 | What is the dustbin capacity? | ✅ | ✅ | 1.00 | 1.00 | product-guide.pdf p.3 |
| q05 | How often should the HEPA filter be replaced? | ✅ | ✅ | 1.00 | 1.00 | product-guide.pdf p.4 |
| q06 | Is there a restocking fee for returns? | ✅ | ✅ | 1.00 | 1.00 | warranty-and-returns.pdf p.1 |
| q07 | How long is the battery covered by warranty? | ✅ | ✅ | 1.00 | 1.00 | warranty-and-returns.pdf p.2 |
| q08 | Where do I find the serial number when creating an RMA? | ✅ | ✅ | 1.00 | 1.00 | warranty-and-returns.pdf p.3 |
| q09 | What should an agent do with an order older than 60 days? | ✅ | ✅ | 1.00 | 1.00 | warranty-and-returns.pdf p.3 |
| q10 | Do you ship to Canada and how long does it take? | ✅ | ✅ | 1.00 | 1.00 | support-faq.pdf p.1 |
| q11 | Does the S7 work with Apple HomeKit? | ✅ | ✅ | 1.00 | 1.00 | support-faq.pdf p.2 |
| q12 | What does Lumora Care+ cost and what does it include? | ✅ | ✅ | 1.00 | 1.00 | support-faq.pdf p.3 |
| q13 | The robot shows error E1. What should the customer do? | ✅ | ✅ | 1.00 | 1.00 | troubleshooting.pdf p.1 |
| q14 | How much clear space does the charging dock need? | ✅ | ✅ | 1.00 | 1.00 | troubleshooting.pdf p.3 |
| q15 | What is the retail price of the Lumora S7? | ✅ | — | n/a | n/a | — |
| q16 | Who is the CEO of Lumora? | ✅ | — | n/a | n/a | — |
| q17 | Does the S7 integrate with Samsung SmartThings? | ✅ | — | n/a | n/a | — |
| q18 | What is the support phone number? | ✅ | — | n/a | n/a | — |
| q19 | Can I return a product after 45 days if it is unopened? | ✅ | ✅ | 1.00 | 1.00 | warranty-and-returns.pdf p.1 |
| q20 | Write me a poem about the ocean. | ✅ | — | n/a | n/a | — |
