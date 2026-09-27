Sixteen test segments picked from `results/qualitative/candidates.jsonl` to cover every qualitative category, with
both wins and losses of the adaptation. "Baseline" is IndicTrans2-1B zero-shot; "Adapted" is IndicTrans2-1B + LoRA.
Texts and sentence-level chrF++ are pulled from the prediction files by `pair_id`, so they cannot drift from what the
systems actually produced. The analyses are my own; corpus-level counts that back the
generalisations are cited from `results/eval/error_analysis.json` and the train split.


## Category coverage

| Category | Examples |
|---|---|
| legal-term | #1, #2, #3, #4, #5, #6, #11, #15, #16 |
| archaic/formulaic | #1, #8, #12, #16 |
| long-sentence | #2, #7, #9 |
| citation/section/date | #2, #7, #10, #11, #12, #14 |
| named-entity | #6, #9, #12, #13, #14, #15 |
| omission/hallucination | #7, #8, #9, #10 |

### Example 1 — archaic/formulaic, legal-term

pair_id: `10-0000` doc_id: `10`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | Leave granted. | |
| **Reference (HI)** | अनुमति प्रदान की गई। | |
| **Baseline (IndicTrans2-1B zero-shot)** | छुट्टी दे दी गई। | 13.7 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | अनुमति प्रदान की गई। | 100.0 |

**Analysis.** "Leave granted" is Supreme Court shorthand for *special leave to appeal is granted*. The baseline reads "leave" as
time off (छुट्टी, "holiday/leave of absence"), a meaning error. The adapter produces the exact formula used in
the Hindi judgments (अनुमति प्रदान की गई), which occurs in 10 training pairs. All three "Leave granted" segments
in the test set (10-0000, 11-0000, 27-0000) flip from wrong or loose to correct. These are the largest single
gains in the test set, but the corpus-level gain survives removing short segments (robustness table in §4).

### Example 2 — legal-term, long-sentence, citation/section/date

pair_id: `27-0019` doc_id: `27`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | As regards other notices it was found that the notice dated 15.04.1974 was sent through registered post as well as through certificate of posting and though there was no evidence of actual service of that notice, the conduct of the Respondents indicated that they were aware of and had knowledge about the notice. | |
| **Reference (HI)** | जैसा कि अन्य नोटिसों के संबंध में यह पाया गया कि दिनांक 15.04.1974 को नोटिस पंजीकृत डाक के साथ-साथ पोस्टिंग के प्रमाण पत्र के माध्यम से भेजा गया था और हालांकि उस नोटिस की वास्तविक तामिल का कोई सबूत नहीं था, प्रत्यर्थिओं के आचरण से दर्शित हुआ कि वे नोटिस के बारे जानते थे और उसका ज्ञान था। | |
| **Baseline (IndicTrans2-1B zero-shot)** | जहां तक अन्य नोटिसों का संबंध है, यह पाया गया कि 15.04.1974 दिनांकित नोटिस पंजीकृत डाक के साथ-साथ पोस्टिंग के प्रमाण पत्र के माध्यम से भेजा गया था और हालांकि उस नोटिस की वास्तविक सेवा का कोई सबूत नहीं था, उत्तरदाताओं के आचरण से संकेत मिलता है कि वे नोटिस के बारे में जानते थे और उन्हें जानकारी थी। | 72.3 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | अन्य नोटिसों के संबंध में यह पाया गया कि दिनांक 15.04.1974 का नोटिस पंजीकृत डाक के साथ-साथ पोस्टिंग के प्रमाण पत्र के माध्यम से भेजा गया था और हालांकि उस नोटिस की वास्तविक सेवा का कोई सबूत नहीं था, प्रत्यर्थीगणों के आचरण से संकेत मिलता है कि वे नोटिस के बारे में जानते थे और उन्हें जानकारी थी। | 80.7 |

**Analysis.** A clear domain fix: the baseline renders "the Respondents" as उत्तरदाताओं ("people who answer a survey"). The
adapter uses the legal term प्रत्यर्थी, although in the double-plural form प्रत्यर्थीगणों (गण + oblique plural;
प्रत्यर्थियों is the correct form; the LLM critic flagged this). It also adopts the reference's date convention (दिनांक 15.04.1974 का
नोटिस instead of 15.04.1974 दिनांकित). What it did **not** learn: "service of notice" stays सेवा ("service" as
in employment/help) in both systems, whereas legal Hindi and the reference use तामील.

### Example 3 — legal-term

pair_id: `11-0020` doc_id: `11`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | The presence of the appellant and the role of the appellant in the incident has emerged from the evidence on record. | |
| **Reference (HI)** | अपीलकर्ता की उपस्थिति और घटना में अपीलकर्ता की भूमिका रिकॉर्ड पर साक्ष्य से स्पष्ट है। | |
| **Baseline (IndicTrans2-1B zero-shot)** | घटना में अपीलार्थी की उपस्थिति और अपीलार्थी की भूमिका अभिलेख पर साक्ष्य से सामने आई है। | 54.3 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | घटना में अपीलकर्ता की उपस्थिति और अपीलकर्ता की भूमिका अभिलेख पर साक्ष्य से सामने आई है। | 73.2 |

**Analysis.** The only change is अपीलार्थी → अपीलकर्ता for "appellant". **Both are correct** Hindi legal terms. The adapter
has simply converged on the majority rendering in the training references (अपीलकर्ता in 195 of 279 appellant
pairs vs अपीलार्थी in 62), and sentence chrF++ jumps +18.8 for a stylistic change. Part of the corpus-level gain
is therefore *house-style matching* rather than better adequacy. That is a real benefit for a court that wants
consistent output, but it should not be read as fewer translation errors.

### Example 4 — legal-term

pair_id: `10-0048` doc_id: `10`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | 15. The necessary orders be issued in the case of the respondent within one month from date the order. | |
| **Reference (HI)** | 15. आदेश की तारीख से एक महीने के भीतर प्रत्यर्थी के मामले में आवश्यक आदेश जारी किए जाएं। | |
| **Baseline (IndicTrans2-1B zero-shot)** | 15. आदेश की तारीख से एक महीने के भीतर प्रत्यर्थी के मामले में आवश्यक आदेश जारी किए जाएं। | 100.0 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | 15. आदेश की तारीख से एक महीने के भीतर प्रतिवादी के मामले में आवश्यक आदेश जारी किए जाएं। | 89.9 |

**Analysis.** A regression. The baseline correctly uses प्रत्यर्थी for the respondent in an appeal; the adapter switches to
प्रतिवादी (the *defendant* in a suit). The adapter learned this ambiguity from the data: of the 128 training pairs
mentioning "respondent", 79 use प्रत्यर्थी but 43 use only प्रतिवादी. Fixing it needs cleaner references or
terminology-constrained decoding (an appeal-context glossary), not more epochs.

### Example 5 — legal-term

pair_id: `27-0010` doc_id: `27`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | 3. Whether the plaintiff’s suit is maintainable? | |
| **Reference (HI)** | 3. क्या वादी का वाद पोषणीय है? | |
| **Baseline (IndicTrans2-1B zero-shot)** | 3. क्या वादी का वाद विचारणीय है? | 74.7 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | 3. क्या वादी का मुकदमा विचारणीय है? | 62.4 |

**Analysis.** "Maintainable" (whether a suit may be entertained at all) has a fixed legal rendering, पोषणीय, which the
reference uses. Both systems produce विचारणीय ("worth considering / triable"), which is close but not the term of
art. The adapter additionally drifts from वाद (the legal word for a civil suit) to the colloquial मुकदमा, a small
register regression. One adaptation epoch over ~1.1k pairs does not overwrite a strongly preferred rendering.

### Example 6 — legal-term, named-entity

pair_id: `11-0009` doc_id: `11`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | The appellant managed to free himself and shot Uma Shanker who fell down as a consequence of a fire arm injury. | |
| **Reference (HI)** | अपीलकर्ता ने खुद को मुक्त करने में कामयाबी हासिल की और उमा शंकर को गोली मार दी जो एक आग्रेय आस्त्र की चोट के परिणामस्वरूप गिर गया और उमा शंकर ने दम तोड़ दिया। | |
| **Baseline (IndicTrans2-1B zero-shot)** | अपीलार्थी स्वयं को मुक्त करने में सफल रहा और उमा शंकर को गोली मार दी, जो हाथ में आग लगने के कारण नीचे गिर गईं। | 35.5 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | अपीलकर्ता खुद को मुक्त करने में कामयाब रहा और उमा शंकर को गोली मार दी, जो आग की बांह की चोट के परिणामस्वरूप नीचे गिर गई। | 55.6 |

**Analysis.** Neither system knows "fire arm" (firearm). The baseline parses it as "fire" + "arm" (हाथ में आग लगने, "his hand
caught fire"). The adapter produces the equally literal आग की बांह की चोट ("injury of the arm of fire"); the
reference points to the right term but misspells it (आग्रेय आस्त्र for आग्नेयास्त्र) and adds a clause that is
not in the source ("उमा शंकर ने दम तोड़ दिया"), so it is not a clean target either. Both systems also inflect the victim Uma Shanker as
feminine (गिर गई/गिर गईं), where the reference correctly uses the masculine गिर गया. The adapter did not touch
these rare forensic terms or the gender of named persons.

### Example 7 — long-sentence, omission/hallucination, citation/section/date

pair_id: `27-0021` doc_id: `27`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | The operative part of its judgment and order dated 19.08.1992 was to the following effect:- “The suit of the plaintiff is decreed for eviction and payment of arrears of rent of Rs.19,561/- with 6% interest per annum pendente lite and the defendant is ordered that they shall vacate the suit property premises no. 3, Khandhari Bazar Lane, as written in para 9 of the suit plaint, within 60 days of this order and otherwise, the court will get it vacated on the expenses of the defendants.” | |
| **Reference (HI)** | दिनांक 19.08.1992 के इस निर्णय और आदेश का ऑपरेटिव भाग निम्नलिखित प्रभाव का था: - " वादी का मुकदमा रु. 19,561/- के किराए के बकाया को मुकदमें के लम्बन के दौरान 6% प्रति वर्ष की दर से ब्याज के साथ भुगतान और निष्कासन के लिए डिक्री किया जाता है और प्रतिवादी को आदेश दिया जाता है कि वे वाद संपत्ति परिसर सं. 3, खंधारी बाजार लेन, जैसा कि वादपत्र के पैरा 9 में लिखा गया है, इस आदेश के 60 दिनों के भीतर खाली कर देंगे और अन्यथा, अदालत प्रतिवादियों के खर्च पर इसे खाली करवाएगी।” | |
| **Baseline (IndicTrans2-1B zero-shot)** | 19.08.1992................................................................................................................................................................................................... | 2.0 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | अपने निर्णय और आदेश दिनांक 19.08.1992 का परिचालन भाग निम्नलिखित प्रभाव के लिए थाः-"वादी का मुकदमा बेदखली और प्रति वर्ष 6 प्रतिशत ब्याज के साथ Rs.19, 561/- के किराए के बकाया के भुगतान के लिए आदेशित किया गया है और प्रतिवादी को आदेश दिया गया है कि वे इस आदेश के 60 दिनों के भीतर वाद संपत्ति परिसर संख्या 3, खंडारी बाजार लेन, जैसा कि वाद वादी के पैरा 9 में लिखा गया है, को खाली कर देंगे और अन्यथा, अदालत इसे प्रतिवादियों के खर्चों पर खाली कर देगी।" | 66.1 |

**Analysis.** The long quoted decree (≈ 90 words, many numbers) makes the **baseline degenerate**: it outputs the date followed
by a run of dots. The adapter produces a complete, well-structured translation and keeps every figure (19.08.1992,
19,561, 6%, premises no. 3, para 9, 60 days). Remaining errors: the Latin *pendente lite* is dropped, "decreed" is
weakened to आदेशित ("ordered"; the reference uses डिक्री), "suit plaint" becomes वाद वादी ("suit plaintiff"; the
reference uses वादपत्र), and "Rs.19,561/-" keeps Latin "Rs." with a spurious space after the comma.

### Example 8 — omission/hallucination, archaic/formulaic

pair_id: `27-0027` doc_id: `27`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | There is nothing on record to prove the service of the said notices. ... ... | |
| **Reference (HI)** | उक्त नोटिसों की तामील को साबित करने के लिए रिकॉर्ड पर कुछ भी नहीं है। ....... | |
| **Baseline (IndicTrans2-1B zero-shot)** | उक्त नोटिसों की सेवा को साबित करने के लिए रिकॉर्ड पर कुछ भी नहीं है। | 78.6 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | उक्त नोटिसों की सेवा को साबित करने के लिए रिकॉर्ड पर कुछ भी नहीं है।....................................................................................................................................................................................... | 67.1 |

**Analysis.** The mirror image of the previous example: here the **adapter degenerates**. After a correct sentence (it keeps
"the said" → उक्त) it appends a long run of dots, apparently triggered by the elision marks "... ..." in the
source (a few training references also contain dot runs). The baseline simply drops the elision. Degenerate
repetition therefore did not disappear with adaptation; it moved (1 flagged segment for each system in
`error_analysis.json`). A decoding-time guard (`no_repeat_ngram_size` or a max-length ratio) would catch both
cases cheaply. Both systems again use सेवा instead of तामील for "service" of a notice.

### Example 9 — long-sentence, omission/hallucination, named-entity

pair_id: `10-0006` doc_id: `10`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | 3. The case of the respondent is also that in identical matters, the appellants have appointed one Balram and one Smt. Geeta Devi, whose father and husband respectively died in harness, as Part Time Tubewell Operators and many other candidates of whose record is not. available. | |
| **Reference (HI)** | 3. प्रत्यर्थी का मामला यह भी है कि अन्य मामलों में, अपीलकर्ताओं ने बलराम और श्रीमती गीता देवी को नियुक्त किया है, जिनके पिता और पति की क्रमशः अंशकालिक ट्यूबवेल ऑपरेटर्स के रूप में कार्य करते हुए मुत्यू हुई और कई अन्य उम्मीदवारों की मृत्यु हो गई, जिनके अभिलेख उपलब्ध नहीं हैं। | |
| **Baseline (IndicTrans2-1B zero-shot)** | 3. प्रत्यर्थी का मामला यह भी है कि समान मामलों में, अपीलकर्ताओं ने एक बलराम और एक श्रीमती गीता देवी को अंशकालिक ट्यूबवेल ऑपरेटर के रूप में नियुक्त किया है, जिनके पिता और पति की क्रमशः मृत्यु हो गई थी और कई अन्य उम्मीदवार जिनके रिकॉर्ड उपलब्ध नहीं हैं। | 73.5 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | 3. प्रत्यर्थी का मामला यह भी है कि समान मामलों में, अपीलकर्ताओं ने एक बलराम और एक श्रीमती गीता देवी को अंशकालिक ट्यूबवेल ऑपरेटरों और कई अन्य उम्मीदवारों के रूप में नियुक्त किया है, जिनके रिकॉर्ड उपलब्ध नहीं हैं। | 63.9 |

**Analysis.** Clause attachment breaks in the adapted output: "appointed … Balram and Smt. Geeta Devi, whose father and husband
respectively died in harness, as Part Time Tubewell Operators and many other candidates…" becomes "appointed …
as Part Time Tubewell Operators and as many other candidates". The clause about the relatives dying in service
is **omitted**, and it is the legal basis of the compassionate-appointment claim. The baseline keeps it (though
"in harness" loses its "while in service" nuance). The reference is itself wrong here (कई अन्य उम्मीदवारों की मृत्यु
हो गई, "many other candidates died"), so chrF++ is a poor judge of this segment.

### Example 10 — citation/section/date, omission/hallucination

pair_id: `27-0007` doc_id: `27`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | Considering the pleadings, following issues were framed on 03.11.1978:- “1. Whether there is a relationship of landlord and tenant between the parties? | |
| **Reference (HI)** | अभिवचन को ध्यान में रखते हुए, निम्नलिखित विवाद्यकों को दिनांक 03.11.1978 को विरचित किया गया था: - "1. क्या पक्षकारों के बीच मकान मालिक और किरायेदार का सम्बन्ध है? | |
| **Baseline (IndicTrans2-1B zero-shot)** | दलीलों पर विचार करते हुए, निम्नलिखित मुद्दों को 03.11.1978 पर तैयार किया गया थाः-"1. क्या पक्षकारों के बीच मकान मालिक और किरायेदार का संबंध है? | 57.4 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | दलीलों पर विचार करते हुए निम्नलिखित मुद्दे तैयार किए गएः-"1. क्या मकान मालिक और किरायेदार का पक्षों के बीच संबंध है? | 37.5 |

**Analysis.** The adapter **drops the date** ("framed on 03.11.1978") altogether, a serious omission in a judgment. The baseline
keeps it but writes "03.11.1978 पर" (spatial "on"); correct Hindi is "दिनांक 03.11.1978 को". This "<date> पर"
pattern appears 7 times in both the baseline and the adapted test output, and never in the references (which use
"को" 13 times). My hypothesis: IndicTransToolkit masks dates as `<ID1>` placeholders before translation, so the
model chooses the postposition without seeing a date. A one-line post-edit rule would fix it; adaptation did not.
Neither system uses the procedural terms of the reference (विवाद्यक विरचित, "issues framed").

### Example 11 — citation/section/date, legal-term

pair_id: `27-0014` doc_id: `27`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | Three more issues were added on 09.04.1981 as under:- “6. Whether the notice of Section 106 T.P. Act, 1882 and under Section 20(2)(a) UP Act No.13 of 1972 was not served on the defendant? | |
| **Reference (HI)** | दिनांक 09.04.1981 को तीन और विवाद्यक इस प्रकार जोड़े गए: - "6. क्या सम्पत्ति अन्तरण अधिनियम की धारा 106 और 1972 की यूपी अधिनियम संख्या 13 के तहत नोटिस प्रतिवादी को नहीं दी गई थी। | |
| **Baseline (IndicTrans2-1B zero-shot)** | 09.04.1981 पर तीन और मुद्दे इस प्रकार जोड़े गएः-"6. क्या प्रतिवादी को टी. पी. अधिनियम, 1882 की धारा 106 और 1972 के यू. पी. अधिनियम की धारा 20 (2) (ए) के तहत नोटिस नहीं दिया गया था? | 51.5 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | 09.04.1981 पर तीन और मुद्दे निम्नानुसार जोड़े गएः-"6. क्या धारा 106 टी. पी. अधिनियम, 1882 और धारा 20 (2) (ए) यूपी अधिनियम No.13 1972 के तहत प्रतिवादी को नोटिस नहीं दिया गया था? | 42.3 |

**Analysis.** Statute references stress word order. The baseline restructures correctly into Hindi genitive order (टी. पी.
अधिनियम, 1882 की धारा 106 … 1972 के यू. पी. अधिनियम की धारा 20 (2) (ए)) but silently drops the Act number
"No. 13". The adapter keeps English order (धारा 106 टी. पी. अधिनियम, 1882 …), leaves "No.13" untranslated in
Latin script and drops the "of" linking it to 1972, which makes the citation harder to parse (all numbers are kept,
so on adequacy it narrowly beats the baseline). The reference expands "T.P. Act" to सम्पत्ति अन्तरण अधिनियम but omits
1882 and Section 20(2)(a), so neither system can match it fully.

### Example 12 — named-entity, citation/section/date, archaic/formulaic

pair_id: `10-0025` doc_id: `10`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | 7. Learned counsel for the appellants however gives an explanation by stating that these persons were originally deployed as Gram Panchayat Vikas Adhikari, transferred and re-transferred back and the challenge to the same was repelled in U.P. Gram Panchayat Adhikari vs Daya Ram Saroj & Ors., (2007) 2 SCC 138. | |
| **Reference (HI)** | 7. अपीलकर्ताओं के विद्गत अधिवक्ता हालांकि यह बताते हुए एक स्पष्टीकरण देते हैं कि इन व्यक्तियों को मूल रूप से ग्राम पंचायत विकास अधिकारी के रूप में तैनात किया विद्गबत था और इन्हें बाद में स्थानांतरित किया था और वापस पुनः स्थानांतरित किया गया था और उसी के लिए चुनौती याचिका को यूपी ग्राम पंचायत अधिकारी बनाम दया राम सरोज और अन्य, (2007) 2 एससीसी 138 में निरस्त कर दिया गया था। | |
| **Baseline (IndicTrans2-1B zero-shot)** | 7. अपीलार्थियों के विद्वान वकील ने हालांकि यह कहते हुए स्पष्टीकरण दिया कि इन व्यक्तियों को मूल रूप से ग्राम पंचायत विकास अधिकारी के रूप में तैनात किया गया था, स्थानांतरित कर दिया गया था और उन्हें वापस स्थानांतरित कर दिया गया था और उसी चुनौती को यू. पी. ग्राम पंचायत अधिकारी बनाम दया राम सरोज और अन्य, (2007) 2 एस. सी. सी. 138 में खारिज कर दिया गया था। | 65.2 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | 7. अपीलकर्ताओं के विद्वान अधिवक्ता हालांकि यह कहते हुए स्पष्टीकरण देते हैं कि इन व्यक्तियों को मूल रूप से ग्राम पंचायत विकास अधिकारी के रूप में तैनात किया गया था, स्थानांतरित कर दिया गया था और वापस स्थानांतरित कर दिया गया था और उसी चुनौती को उत्तर प्रदेश ग्राम पंचायत अधिकारी बनाम दया राम सरोज और अन्य, (2007) 2 एससीसी 138 में खारिज कर दिया गया था। | 74.8 |

**Analysis.** Several register wins for the adapter: "learned counsel" → विद्वान अधिवक्ता (legal register) instead of विद्वान वकील
(colloquial "lawyer"); present tense matching the source ("gives an explanation", देते हैं); and the reference's
citation style "(2007) 2 एससीसी 138" instead of the spelled-out "एस. सी. सी.". One named-entity regression: inside
the case title "U.P. Gram Panchayat Adhikari vs Daya Ram Saroj" it expands U.P. to उत्तर प्रदेश, but case titles
should be reproduced as cited (the reference keeps यूपी).

### Example 13 — named-entity

pair_id: `11-0023` doc_id: `11`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | This was objected to by the complainant Gaya Prasad (PW 1). | |
| **Reference (HI)** | शिकायतकर्ता गया प्रसाद (प्र. सा. 1) द्वारा इस पर आपत्ति जताई गई थी। | |
| **Baseline (IndicTrans2-1B zero-shot)** | शिकायतकर्ता गया प्रसाद (पीडब्लू 1) ने इसका विरोध किया था। | 39.1 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | इस पर शिकायतकर्ता गया प्रसाद (पीडब्लू 1) ने आपत्ति जताई थी। | 57.3 |

**Analysis.** The adapter re-orders the clause to the reference's structure (इस पर … आपत्ति जताई), hence +18 chrF++. Both
systems transliterate the witness label "PW 1" as पीडब्लू 1, while the reference uses the Hindi abbreviation
"प्र. सा. 1". Such abbreviations are rare and inconsistent in the training data, so the adapter did not pick them up.

### Example 14 — citation/section/date, named-entity

pair_id: `10-0021` doc_id: `10`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | The respondent has stated that during the period of 13 years of the employment of the father of the respondent, he was transferred from the Irrigation Department to Panchayati Raj Department as ‘Gram Panchayat Vikas Adhikari’ and vice versa, i.e, he was transferred 2 or 3 times by the appellants and was even appointed as a polling officer by State Election Commission on 15.10.2015, 26.11.2015 and 03.12.2015. | |
| **Reference (HI)** | प्रत्यर्थी ने कहा है कि प्रत्यर्थी के पिता के रोजगार के 43 वर्षों की अवधि के दौरान, उन्हें सिंचाई विभाग से पंचायती राज विभाग में 'ग्राम पंचायत विकास अधिकारी' के रूप में स्थानांतरित किया गया था और इसके विपरीत क्रम में ,यानी, उन्हें अपीलकर्ताओं द्वारा 2 या 3 बार स्थानांतरित किया गया था और यहां तक कि 15.10.2015, 26.11.2015 और 03.12.2015 को राज्य निर्वाचन आयोग द्वारा मतदान अधिकारी के रूप में नियुक्त किया गया था। | |
| **Baseline (IndicTrans2-1B zero-shot)** | प्रत्यर्थी ने कहा है कि प्रत्यर्थी के पिता की नौकरी के 13 वर्षों की अवधि के दौरान, उन्हें सिंचाई विभाग से पंचायती राज विभाग में'ग्राम पंचायत विकास अधिकारी'के रूप में स्थानांतरित किया गया था और इसके विपरीत, यानी, उन्हें अपीलकर्ताओं द्वारा 2 या 3 बार स्थानांतरित किया गया था और यहां तक कि राज्य चुनाव आयोग द्वारा 15.10.2015, 26.11.2015 और 03.12.2015 पर एक मतदान अधिकारी के रूप में नियुक्त किया गया था। | 85.8 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | प्रत्यर्थी ने कहा है कि प्रत्यर्थी के पिता की नौकरी के 13 वर्षों की अवधि के दौरान, उसे सिंचाई विभाग से पंचायती राज विभाग में'ग्राम पंचायत विकास अधिकारी'के रूप में स्थानांतरित किया गया था और इसके विपरीत, यानी, उसे अपीलकर्ताओं द्वारा 2 या 3 बार स्थानांतरित किया गया था और यहां तक कि राज्य चुनाव आयोग द्वारा 15.10.2015, 26.11.2015 और 03.12.2015 पर एक मतदान अधिकारी के रूप में नियुक्त किया गया था। | 81.7 |

**Analysis.** **Reference error.** The source says "13 years"; the reference says 43 वर्षों. This is the same legacy-font 1→4
digit corruption I had to repair in the OCR'd training documents, here inside an official test reference. Both
systems correctly write 13 and are penalised for it. The adapter also changes the honorific उन्हें to the plain उसे
for the respondent's deceased father (the reference keeps उन्हें), and both systems use the "<date> पर" pattern
discussed in 27-0007.

### Example 15 — named-entity, legal-term

pair_id: `10-0003` doc_id: `10`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | The High Court in the impugned CRaneet ee . : aaa respondent was unambiguous and against the regular Reason: a. opined that the selection process of the father of vacancy whereby he had _ submitted all the _ requisite documents to the Irrigation Department. | |
| **Reference (HI)** | उच्च न्यायालय के आशक्षिपित आदेश में कहा गया कि प्रत्यर्थी के पिता की चयन प्रक्रिया अस्पष्ट थी और नियमित रिक्ति के खिलाफ थी, जिसके तहत उन्होंने सिंचाई विभाग को सभी अपेक्षित दस्तावेज प्रस्तुत किए थे। | |
| **Baseline (IndicTrans2-1B zero-shot)** | उच्च न्यायालय ने विवादित सी. आर. ए. एन. ई. ई.: ए. ए. ए. प्रतिवादी में स्पष्ट और नियमित कारण के खिलाफ थाः ए. राय दी कि रिक्ति के पिता की चयन प्रक्रिया जिसके तहत उन्होंने सिंचाई विभाग को सभी आवश्यक दस्तावेज प्रस्तुत किए थे। | 60.6 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | उच्च न्यायालय ने आक्षेपित सी. आर. ए. एन. ई.: एएए प्रतिवादी स्पष्ट और नियमित कारण के खिलाफ थाः ए. राय दी कि रिक्ति के पिता की चयन प्रक्रिया जिसके द्वारा उन्होंने सिंचाई विभाग को सभी आवश्यक दस्तावेज प्रस्तुत किए थे। | 57.8 |

**Analysis.** **Source-side noise.** The English "clean" text of this sentence is OCR debris ("impugned CRaneet ee . : aaa
respondent was unambiguous…"), while the Hindi reference was translated from the original judgment. Both
systems faithfully transliterate the garbage (सी. आर. ए. एन. ई. ई.). No MT adaptation helps here; the fix is
source-side OCR checking (e.g. flagging tokens outside an English vocabulary before translation). The adapter
does get "impugned" right (आक्षेपित, the legal term) where the baseline has विवादित ("disputed"). The reference
says the selection process was अस्पष्ट ("unclear"), while the garbled English says "unambiguous"; I cannot tell
which is right without the original PDF.

### Example 16 — legal-term, archaic/formulaic

pair_id: `27-0003` doc_id: `27`

| | Text | chrF++ |
|---|---|---|
| **Source (EN)** | This notice was admittedly received by the Respondents. | |
| **Reference (HI)** | यह नोटिस प्रत्यर्थिओं द्वारा निःसन्देह रूप से प्राप्त की गयी था। | |
| **Baseline (IndicTrans2-1B zero-shot)** | यह सूचना प्रत्यर्थियों द्वारा स्वीकार की गई थी। | 27.8 |
| **Adapted (IndicTrans2-1B + LoRA (legal))** | यह नोटिस प्रत्यर्थीगणों को स्वीकार्य रूप से प्राप्त हुआ था। | 49.9 |

**Analysis.** "This notice was admittedly received": the baseline's स्वीकार की गई थी means "was *accepted*", which changes a
finding about receipt into one about acceptance, a legally material error. The adapter keeps the meaning
"received" (प्राप्त हुआ) and matches the reference's नोटिस, but renders the formulaic "admittedly" (= undisputedly)
as स्वीकार्य रूप से ("acceptably"; स्वीकृत रूप से would be closer). The reference (निःसन्देह रूप से … की गयी था) is
itself loose and ungrammatical.

## Second opinion (LLM judgement — not a metric)

An LLM reviewer (prompted as an English–Hindi legal translator) reviewed the 16 examples
**blind to my analyses** (it read `curated_examples.jsonl` first and `examples.md` only afterwards). Its
verdicts on these hand-picked items, which were chosen to include adaptation losses, so they say nothing about the
overall rate: adapted better in 7, baseline better in 6, ties in 3.

Its cross-cutting points:
1. **Fixed:** high-frequency formulas and party labels ("अनुमति प्रदान की गई", उत्तरदाताओं → प्रत्यर्थी, वकील → अधिवक्ता,
   विवादित → आक्षेपित, "दिनांक … का", "एससीसी"), plus one important adequacy fix ("accepted" → "received", 27-0003).
2. **Broke:** consistency and honorifics (प्रत्यर्थी → प्रतिवादी, the double plural प्रत्यर्थीगणों, उन्हें → उसे, वाद → मुकदमा),
   learned from inconsistent references; and structure in long sentences (a dropped date and a dropped clause,
   English order in statute citations).
3. **Untouched:** सेवा for तामील, मुद्दे/दलीलें for विवाद्यक/अभिवचन, विचारणीय for पोषणीय, "<date> पर", the visarga "ः" used
   as a colon, पीडब्लू, "firearm", the victim's gender. Dot-run degeneration appears in both systems, pointing to
   a decoding guard rather than more adaptation.
4. **Reference quality:** it judged 9 of the 16 references loose or wrong, two of them meaning-changing
   ("43 वर्षों"; an invented death clause), which limits any reference-based score on this test set.

Where it disagreed with me, the analyses above were corrected (double plural in 27-0019; reference
problems in 11-0009, 10-0003, 27-0003; the exact loss in 27-0014).

*This is LLM judgement on 16 hand-picked items, not a metric.*

