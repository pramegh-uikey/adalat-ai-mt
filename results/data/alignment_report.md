# Alignment report

## Method

Hard-wrapped text is unwrapped into segments, cleaned (BOM/zero-width removal, EN `li.`→`ii.` OCR fix, HI date-space and double-danda fixes) and split into sentences with rule-based EN/HI splitters. Each document is then aligned with a monotonic dynamic programme over LaBSE embeddings of 1–3-sentence blocks (moves 1-1, 1-2, 2-1, 1-3, 3-1 and skips), with a soft paragraph-number anchor bonus. Pairs below the score threshold or outside the length-ratio band are dropped. Config: `{'max_block': 3, 'skip_threshold': 0.4, 'merge_penalty': 0.05, 'anchor_bonus': 0.05, 'min_score': 0.6, 'len_ratio': (0.5, 2.5)}`.

**Hindi text recovered from PDF (OCR + text-layer numbers):** docs 6, 14, 22, 25, 26 — the provided Hindi clean text is entirely '?'; recovered text is OCR output (train split only).

## Pairs per document

| doc_id | split | n_en | n_hi | kept | dropped |
|---|---|---|---|---|---|
| 1 | train | 67 | 65 | 63 | 1 |
| 2 | train | 25 | 29 | 24 | 0 |
| 3 | train | 32 | 34 | 31 | 0 |
| 4 | train | 55 | 54 | 53 | 0 |
| 5 | train | 41 | 42 | 39 | 0 |
| 6 | train | 36 | 37 | 36 | 0 |
| 7 | dev | 54 | 58 | 52 | 1 |
| 8 | train | 42 | 41 | 41 | 0 |
| 9 | dev | 25 | 25 | 24 | 0 |
| 10 | test | 55 | 49 | 49 | 0 |
| 11 | test | 37 | 32 | 29 | 0 |
| 12 | train | 65 | 66 | 65 | 0 |
| 13 | train | 35 | 36 | 28 | 2 |
| 14 | train | 39 | 35 | 29 | 1 |
| 15 | train | 54 | 50 | 49 | 0 |
| 16 | train | 47 | 49 | 47 | 0 |
| 17 | train | 32 | 32 | 31 | 0 |
| 18 | train | 41 | 39 | 39 | 0 |
| 19 | train | 49 | 47 | 47 | 0 |
| 20 | train | 58 | 57 | 50 | 2 |
| 21 | train | 72 | 71 | 71 | 0 |
| 22 | train | 48 | 68 | 47 | 1 |
| 23 | train | 92 | 91 | 89 | 0 |
| 24 | train | 87 | 84 | 77 | 2 |
| 25 | train | 83 | 86 | 77 | 0 |
| 26 | train | 41 | 39 | 37 | 0 |
| 27 | test | 45 | 45 | 42 | 0 |
| 28 | dev | 96 | 86 | 86 | 0 |
| 29 | train | 34 | 34 | 34 | 0 |
| 30 | train | 48 | 48 | 45 | 1 |

## Score distribution

Quantiles: p5=0.7554, p25=0.8339, p50=0.8714, p75=0.9017, p95=0.9324

```
[0.0,0.1)     0 
[0.1,0.2)     0 
[0.2,0.3)     0 
[0.3,0.4)     0 
[0.4,0.5)     2 #
[0.5,0.6)     5 #
[0.6,0.7)    24 #
[0.7,0.8)   153 #######
[0.8,0.9)   878 ########################################
[0.9,1.0)   380 #################
```

## Drop / merge rates

- Drop rate: 0.0076 (by reason: {'low_score': 7, 'len_ratio': 4})
- Merge rate (non 1-1 pairs): 0.0513 (by type: {'1-1': 1368, '2-1': 35, '1-2': 33, '1-3': 1, '3-1': 5})
- Pairs per split: {'train': 1149, 'dev': 162, 'test': 120}

## Audit sample (random kept pairs, all splits)

Judged by reading: 20/20; precision (correct) = 19/20 = 0.950 (95% Wilson CI 0.764–0.991); lenient (correct + partial) = 1.000

| pair_id | score | verdict | note | en | hi |
|---|---|---|---|---|---|
| 13-0024 | 0.8936 | correct | free translation, same content | 17. The case is remanded to the High Court for deciding the applications filed by the respondents(accused persons) for grant of bail/suspension of their jail sentence afresh on their respective merits keeping in view the law laid down by this Court in the aforementioned decisions coupled with the material filed by the State and the complainant in support of their case. | 17. इस मामले को प्रत्यर्थियों (अभियुक्तों) द्वारा दायर जमानत/कारावास दण्ड के निलंबन के लिए दायर किए गए आवेदनों को तय करने के लिए मा. उच्च न्यायालय ने प्रतिप्रेषित किया है, जो कि इस न्यायालय द्वारा पूर्व में घोषित किए गए विधि को ध्यान में रखते हुए अपने संबंधित गुण -दोषों के आधार पर राज्य और शिकायतकर्ता द्वारा उनके मामले के समर्थन में दायर की गई सामग्री के साथ जुड़ा है। |
| 15-0032 | 0.7348 | correct |  | It is further submitted that there is no valid delegation to Cantonment Executive Officer at all and the impugned notices are issued without any jurisdiction. | आगे यह तर्क दिया गया कि छावनी कार्यकारी अधिकारी का किसी भी तरह का विधिमान्य प्रतिनिधि मंडल नहीं है और आशक्षैपित सूचनाएँ बिना किसी न्यायक्षेत्र के जारी की गयी है। |
| 30-0017 | 0.8411 | correct |  | An analysis of the said provision, in our opinion, is essential to adjudicate upon the present case. | हमारी राय में, वर्तमान मामले पर निर्णय लेने के लिए उक्त प्रावधान का विश्लेषण आवश्यक है। |
| 30-0014 | 0.8919 | correct | 'cross objections' loosely rendered as आपत्तियों | 11. To appreciate the issue at hand and to come to a correct conclusion, we must analyze the impugned order passed by the High Court, to see whether there has been any application of mind by the High Court on the cross objections filed by the appellants herein, and if such consideration has not been taken into account, then to what extent can this Court grant relief. | 11. मौजूदा मुद्दे की समझने और सही निष्कर्ष पर पहुंचने के लिए, हमें उच्च न्यायालय द्वारा पारित आदेश का विश्लेषण यह देखने के लिए करना चाहिए कि क्या यहां अपीलकर्ताओं द्वारा दायर की गई आपत्तियों पर उच्च न्यायालय द्वारा विचार किया गया है, और यदि इस तरह के विचार को ध्यान में नहीं रखा गया है, तो यह न्यायालय किस हद तक राहत दे सकता है। |
| 10-0017 | 0.8264 | correct |  | The regular vacancy was held to mean a vacancy which occurred in posts sanctioned by the competent authority. | नियमित रिक्ति का अर्थ एक रिक्ति से था जो सक्षम प्राधिकारी द्वारा स्वीकृत पदों में हुई थी। |
| 28-0069 | 0.8923 | correct |  | If that be so there was also no lease in place as on the date of the application namely 11.11.2011. | यदि ऐसा है तो आवेदन की तारीख के अनुसार दिनांक 11.11.2011 को कोई पट्टा नहीं था। |
| 12-0031 | 0.8780 | correct |  | PW-2, Phoola was also an eye witness to the offence. | पीडब्ल्यू-2, फूला भी इस अपराध की चश्मदीद गवाह थी। |
| 29-0012 | 0.9006 | correct |  | It is submitted that the Reference Court in that case relied upon the sale deed exemplar of the year 1978 and thereafter determined the market value of the compensation at Rs. 15,402/- per acre. | यह कहा गया है कि उस मामले में संदर्भ न्यायालय ने वर्ष 1978 के बिक्री विलेख पर अवलम्ब लिया और उसके बाद मुआवजे का बाजार मूल्य 15,402/- रूपये प्रति एकड़ पर निर्धारित किया। |
| 8-0003 | 0.9191 | correct | Devanagari digit ३ in the reference | 3.1 From the impugned judgment and order passed by the High Court and even taking into consideration the counter affidavit filed on behalf of the respondent nos. 2 and 3 - Collector and the Lucknow Development Authority filed before the High Court, it appears that it was the specific case on behalf of the appellant - Authority that the possession of the land in question was duly taken on 13.02.2003 by the Special Land Acquisition Officer and was delivered to the Lucknow Development Authority vide Possession Certificate dated 13.02.2003. | 3.1 उच्च न्यायालय द्वारा पारित आक्षेपित निर्णय और आदेश से और प्रतिवादी संख्य- 2 और ३ -कलेक्टर और लखनऊ विकास प्राधिकरण की ओर से उच्च न्यायालय में दाखिल जवाबी शपथ पत्र पर भी विचार करते हुए, यह प्रतीत होता है कि यह अपीलकर्ता-प्राधिकरण की ओर से विशेष मामला था कि प्रश्नगत भूमि का कब्जा विशेष भूमि अधिग्रहण अधिकारी द्वारा 13.02.2003 को विधिवत लिया गया था और 13.02.2003 को लखनऊ विकास प्राधिकरण को सौंपा गया था। |
| 12-0020 | 0.9196 | correct |  | PW- 2 was cousin sister of Mullu, PW-1, as well as Malkhan. | पीडब्लू-2, पीडब्लू-1 मुल्लू के साथ-साथ मलखान की चचेरी बहन थी। |
| 28-0046 | 0.9221 | correct | quoted statutory text (s.105 TP Act) aligned as one unit | 12. Section 105 specifically deals with lease of immovable property, and it reads as follows: - “105. Lease defined- A lease of immovable property is a transfer of a right to enjoy such property, made for a certain time, express or implied, or in perpetuity, in consideration of a price paid or promised, or of money, a share of crops, service or any other thing of value, to be rendered periodically or on specified occasions to the transferor by the transferee, who accepts’ the transfer on such terms.” | 12. धारा 105 विशेष रूप से अचल संपत्ति के पट्टे से संबंधित है, और यह इस प्रकार है:- “105. स्थावर सम्पत्ति का पट्टा ऐसी संपत्ति का उपभोग करने के अधिकार का ऐसा अन्तरण है, जो एक अभिव्यक्त या विवक्षित समय के लिए या शाश्वत काल के लिए किसी कीमत के, जो दी गई हो या जिसे देने का वचन दिया गया हो, अथवा धन, या फसलों के अंश या सेवा या किसी मूल्यवान वस्तु के, जो कालावधीय रूप से या विनिर्दिष्ट अवसरों पर अन्तरिती द्वारा, जो उस अन्तरण को ऐसे निबंधनों पर प्रतिगृहीत करता है, अन्तरक को की या दी जानी है, प्रतिफल के रूप में किया गया हो।” |
| 10-0020 | 0.8767 | correct |  | 6. We may note an interesting aspect pointed out by the learned counsel for the respondent, inter alia, in his synopsis (as usual the appellants did not consider it appropriate to assist this Court by filing a synopsis as had been directed vide the last order, apart from the note on the cause list!). | 6. हम अन्य बातों के साथ प्रत्यर्थी के विद्वत अधिवक्ता द्वारा अपने सारांश में इंगित एक दिलचस्प पहलू पर ध्यान दे सकते हैं (सामान्यतया अपीलकर्ताओं ने एक प्रारूप दाखिल करके इस न्यायालय की सहायता करना उचित नहीं समझा, जैसा कि अंतिम आदेश के अनुसार निर्देशित किया था, काज लिस्ट मे सूचना के अलावा!)। |
| 7-0020 | 0.8758 | correct |  | Section 331 of the Act has been the subject of series of pronouncements of the High Court as to the circumstances and the nature of the suits in which its exclusionary effect operates. | अधिनियम की धारा 331, उच्च न्यायालय के विभिन्न घोषणाओं की श्रृंखला में ऐसी परिस्थितियों एवं वाद की प्रकृति के संबंध में जिसमें इसका बहिष्करण प्रभाव संचालित होता है, का विषय रही है। |
| 4-0027 | 0.7841 | correct |  | The appellant continues to remain in service and his salary has been paid. | अपीलार्थी अपनी सेवा में है और उसके वेतन का भुगतान कर दिया गया है। |
| 24-0055 | 0.8076 | correct |  | Take for instance the case on hand. | उदाहरण के लिए वर्तमान मामले को ही लें। |
| 11-0026 | 0.8512 | partial | HI writes 'भाग-I' with a danda (भाग-।), so the HI sentence was split there; its tail is missing | 7 Having regard to the circumstances of the case, we are of the view that the conviction under Section 302 of the IPC should be converted to one under Section 304 Part \|. | 7. मामले की परिस्थितियों को ध्यान में रखते हुए, हमारा विचार है कि भारतीय दंड संहिता की धारा 302 के अधीन दोषसिद्धि को धारा 304 भाग-। |
| 15-0040 | 0.9384 | correct | reference is loose but content-parallel | So far as Afzal who is respondent in Civil Appeal No. 3814 of 2019 is concerned, show cause notice dated 22.08.2006 is issued alleging that he has constructed the shop no.53-54 at Ghosi Mohalla, B.I. Bazar, Meerut Cantt., but same is not even referred to in the final notice issued on 02.09.2006. | जहाँ तक अफजल, जो कि सिविल अपील सं. 3814/2019 का प्रत्यर्थी है, का सम्बन्ध है दिनांक 22.08.2006 को आरोपित करते हुए कारण बताओ नोटिस जारी किया गया कि उसने दुकान सं. 53-54 घोसी मोहल्ला, बी. आर. बाजार मेरठ कैन्ट में निर्माण कार्य कराया है, लेकिन एकरूप नहीं है, यहाँ तक कि दिनांक 02.09.2006 को जारी अंतिम नोटिस में निर्दिष्ट नहीं है। |
| 1-0061 | 0.8649 | correct |  | 13. Following the aforementioned judgment, we are of the opinion that the decision of the Disciplinary Authority in not paying the salary for the period of suspension cannot be said to be contrary to law. | 13. उपर्युक्त निर्णय के बाद, हमारा विचार है कि निलंबन की अवधि के लिए वेतन का भुगतान नहीं करने के अनुशासनात्मक प्राधिकरण के निर्णय को विधि विरूद्ध नहीं कहा जा सकता है। |
| 21-0057 | 0.8122 | correct | multi-level marker 21.8. (viii) preserved on both sides | 21.8. (viii) There is a distinction between inordinate delay and a delay of short duration or few days, for to the former doctrine of prejudice is attracted whereas to the latter it may not be attracted. | 21.8. (viii) अत्यधिक देरी और कम अवधि या कुछ दिनों की देरी के बीच अंतर है, क्योंकि पहले वाले में पूर्वाग्रह का सिद्धांत लागू होता है जबकि दूसरे में ऐसा नहीं हो सकता है। |
| 7-0012 | 0.8245 | correct |  | 5. We have heard the learned counsel for the parties. | 5. हमने पक्षकारों के विद्बत अधिवक्ता को सुना है। |

## Supplementary audit (OCR-recovered Hindi documents)

Judged by reading: 10/10; precision (correct) = 9/10 = 0.900 (95% Wilson CI 0.596–0.982); lenient (correct + partial) = 1.000

| pair_id | score | verdict | note | en | hi |
|---|---|---|---|---|---|
| 25-0000 | 0.6646 | correct | HI numbers 'Leave granted' as para 1, EN does not | Leave granted. | 1. अनुमति प्रदान की गई। |
| 26-0029 | 0.8953 | correct |  | 9. In the present case, the findings based on the expert’s evidence are that the base paint was mixed with colouring as an additive. | 9. वर्तमान मामले में, विशेषज्ञ के साक्ष्य के आधार पर निष्कर्ष यह है कि बेस पेंट को एक संयोजक के रूप में रंग के साथ मिलाया गया था। |
| 22-0023 | 0.8483 | correct |  | In our view, the demise of the appellant’s father would not negate the right which stood vested in the appellant. | हमारे विचार में, अपीलकर्ता के पिता के निधन से अपीलकर्ता को प्राप्त अधिकार समाप्त नहीं होगा। |
| 26-0017 | 0.7626 | correct | free translation | This court held that the process involved mixing crushed coal with suitable binders pressed in briquetting press, from which regular shaped briquettes were suitably carbonised. | इस न्यायालय ने अवधारित किया कि इस प्रक्रिया में मिक्स किये हुए कोयले को ब्रिकेटिंग प्रेस में अच्छी तरह दबाकर और उपयुक्त तरीके से बाँधकर मिलाया जाता था, जिससे नियमित आकार के ब्रिकेट को उपयुक्त रूप से कार्बनीकृत किया जाता था। |
| 22-0034 | 0.8823 | correct | date 23.11.2009 restored by text-layer number repair (OCR gave 23..2009) | By way of letters dated 23.11.2009 and 18.10.2010, the appellant requested NOIDA to move forward with the allotment. | दिनांक 23.11.2009 और 18.10.2010 के पत्रों के माध्यम से, अपीलकर्ता ने नोएडा से आवंटन के साथ आगे बढ़ने का अनुरोध किया। |
| 25-0001 | 0.8386 | partial | HI block also contains the following list of SLP numbers | These four criminal appeals have been preferred by the common Appellant (original complainant) against four separate orders of the High Court of Judicature at Allahabad (Lucknow Bench), granting bail to the respective Respondent No.2 in each of the following Special Leave Petitions: | 2. ये चार आपराधिक अपीलें उभयनिष्ठ अपीलकर्ता (मूल परिवादी) द्वारा इलाहाबाद उच्च न्यायालय (लखनऊ पीठ) के चार अलग-अलग आदेशों के विरुद्ध दायर की गई हैं, जिसमें निम्नलिखित विशेष अनुमति याचिकाओं में से प्रत्येक में संबंधित प्रतिवादी सं. 2 को जमानत प्रदान की गई है: * एसएलपी (क्रि.) सं. 015156/2024 (प्रत्यर्थी सं. 2: मूल अभियुक्त सं. 3, श्रीमती तारा बानो, मृतका की सास), * एसएलपी (क्रि.) सं. 11355/2024 (प्रत्यर्थी सं. 2: मूल अभियुक्त सं. 2, मुख्तार अहमद, मृतका के ससुर), * एसएलपी (क्रि.) सं. 015157/2024 (प्रत्यर्थी सं. 2: मूल अभियुक्त सं. 5, आयशा खान, मृतका की ननद), * एसएलपी (क्रि.) सं. 015158/2024 (प्रत्यर्थी सं. 2: मूल अभियुक्त संख्या 4, सबा, मृतका की ननद)। |
| 25-0010 | 0.8966 | correct |  | The FIR further recounts that on 22.01.2024, around 6:15 p.m., the father of the Appellant received a phone call from Accused No.2 (Mukhtar Ahmad/father-in-law) asking him to come immediately. | 5. एफआईआर में आगे बताया गया है कि 22.01.2024 को शाम करीब 6:15 बजे अपीलकर्ता के पिता को आरोपी सं. 2 (मुख्तार अहमद/ससुर) का फोन आया जिसमें उसे तुरंत आने के लिए कहा गया। |
| 26-0001 | 0.8563 | correct |  | 2. The facts in all these cases are that the assessees are dealers in inter alia paints. | 2. अन्य बातों के साथ-साथ इन सभी मामलों में तथ्य यह है कि निर्धारिती पेंट के व्यापारी होते हैं। |
| 22-0001 | 0.7245 | correct |  | Leave granted. | अनुमति प्रदान की गई। |
| 25-0005 | 0.9433 | correct |  | It states that Shahida was married on 07.02.2022 to Accused No.1, Sami Khan (husband of the deceased). | इसमें कहा गया है कि शाहिदा की शादी 07.02.2022 को आरोपी सं. 1, समी खान (मृतका का पति) से हुई थी। |
