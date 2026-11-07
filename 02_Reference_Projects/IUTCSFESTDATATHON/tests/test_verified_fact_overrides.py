from pathlib import Path
import unittest

from verified_fact_overrides import load_verified_entries, predict_verified


class VerifiedFactOverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entries = load_verified_entries(Path("resources/verified_fact_answers.json"))

    def test_every_loaded_entry_has_two_independent_sources(self) -> None:
        self.assertTrue(self.entries)
        for entry in self.entries.values():
            self.assertGreaterEqual(len(set(entry["sources"])), 2)

    def test_birbal_relation(self) -> None:
        entry = next(item for item in self.entries.values() if "বীরবল" in item["question"])
        self.assertEqual(predict_verified("প্রমথ চৌধুরী", entry), 1)
        self.assertEqual(predict_verified("ধূর্জটি প্রসাদ মুখোপাধ্যায়", entry), 0)

    def test_stone_age_sequence(self) -> None:
        entry = next(item for item in self.entries.values() if "পুরাপ্রস্তর" in item["question"])
        self.assertEqual(predict_verified("নব্যপ্রস্তর", entry), 1)
        self.assertEqual(predict_verified("মধ্যপ্রস্তর যুগ", entry), 0)

    def test_recovered_bcs_lineage_answers(self) -> None:
        by_question = {item["question"]: item for item in self.entries.values()}
        poem = by_question["ভাষা আন্দোলন নিয়ে লেখা কবিতার নাম বলো একটি।"]
        self.assertEqual(predict_verified("স্মৃতিস্তম্ভ", poem), 1)
        self.assertEqual(predict_verified("তোমাকে অভিবাদন প্রিয়া", poem), 0)
        energy = by_question["একটি নবায়নযোগ্য জ্বালানীর নাম বলো।"]
        self.assertEqual(predict_verified("পরমাণু শক্তি", energy), 1)
        self.assertEqual(predict_verified("কয়লা", energy), 0)
        flight = by_question[
            "ঢাকা থেকে হংকং হয়ে প্লেন নিউইয়র্কে যাওয়ার সময় দিনের সময়কালকে অপেক্ষাকৃত ছোট মনে হয়, কেন?"
        ]
        self.assertEqual(predict_verified("অন্য কোন কারণ আছে", flight), 1)
        self.assertEqual(predict_verified("পৃথিবী সূর্যের চারদিকে ঘুরছে বলে", flight), 0)

    def test_shakib_birth_date_repair(self) -> None:
        by_question = {item["question"]: item for item in self.entries.values()}
        birth = by_question["সাকিব আল হাসান কত সালে জন্মগ্রহণ করেন?"]
        self.assertEqual(predict_verified("২৪ মার্চ ১৯৮৭", birth), 1)
        self.assertEqual(predict_verified("২৩ মার্চ ১৯৮৭", birth), 0)

    def test_pather_panchali_part_chapter_swap(self) -> None:
        by_question = {item["question"]: item for item in self.entries.values()}
        entry = by_question["বিভূতিভূষণ বন্দ্যোপাধ্যায়ের 'পথের পাঁচালী' উপন্যাসটি মোট কয়টি খণ্ডে বিভক্ত?"]
        self.assertEqual(predict_verified("তিনটি খণ্ড", entry), 1)
        self.assertEqual(predict_verified("পঁয়ত্রিশটি খণ্ড", entry), 0)

    def test_verified_antonym_alternatives(self) -> None:
        by_question = {item["question"]: item for item in self.entries.values()}
        expected = {
            "তামসিক' শব্দের শুদ্ধ বিপরীত শব্দ কোনটি?": "রাজসিক",
            "কদাচিৎ' শব্দের শুদ্ধ বিপরীত শব্দ কোনটি?": "প্রায়শ",
            "সাকার' শব্দের শুদ্ধ বিপরীত শব্দ কোনটি?": "অরূপ",
            "শিষ্ট' শব্দের শুদ্ধ বিপরীত শব্দ কোনটি?": "বেয়াদব",
            "স্বাধীন' শব্দের শুদ্ধ বিপরীত শব্দ কোনটি?": "বন্দী",
        }
        for question, answer in expected.items():
            self.assertEqual(predict_verified(answer, by_question[question]), 1)

    def test_verified_relation_swaps(self) -> None:
        by_question = {item["question"]: item for item in self.entries.values()}
        cases = [
            ("অপারেশন সার্চলাইট কবে শুরু হয়?", "১৭ এপ্রিল ১৯৭১ সালে", "২৫ মার্চ ১৯৭১"),
            ("মেঘনা নদীর উৎস কোন নদী?", "গঙ্গা নদী", "বরাক নদী"),
            ("শেষের কবিতা উপন্যাসের ধারাবাহিক প্রকাশ কত সালে শুরু হয়?", "১৯২৮ সালে", "১৯২৭ সালে"),
            ("বাংলাদেশ কাজী নজরুল ইসলামকে জাতীয় কবির মর্যাদা কত সালে দেয়?", "১৯৭৪ সালে", "১৯৭২ সালে"),
        ]
        for question, wrong, correct in cases:
            self.assertEqual(predict_verified(wrong, by_question[question]), 0)
            self.assertEqual(predict_verified(correct, by_question[question]), 1)

    def test_verified_science_and_geography_confirmations(self) -> None:
        by_question = {item["question"]: item for item in self.entries.values()}
        cases = [
            ("সুইডেন দেশটি Group of Seven(G-7) এর সদস্য?", "না"),
            ("প্রাকৃতিক ইউরেনিয়ামে শতকরা কতভাগ 238U আইসোটোপ থাকে?", "99.30%"),
            ("ভারতীয় নাগাল্যান্ড রাজ্যের সাথে বাংলাদেশের কোনো সীমান্ত নেই। এটি কি সত্য?", "সত্য"),
            ("যমুনা সেতুর দৈর্ঘ্য‒", "৪.৮ কিলোমিটার"),
        ]
        for question, answer in cases:
            self.assertEqual(predict_verified(answer, by_question[question]), 1)

        self.assertEqual(
            predict_verified("রেডিও ফ্রিকুয়েন্সি আইডেন্টিফিকেশন", by_question["RFID বলতে বােঝায়-"]),
            1,
        )
        self.assertEqual(
            predict_verified("লৌহ", by_question["তেজস্ক্রীয় পদার্থ নয় এমন একটি ধাতুর নাম বলো।"]),
            1,
        )

        history_cases = [
            ("বাংলাদেশের স্বাধীনতা দিবস কত তারিখে?", "২৬ মার্চ"),
            ("বাংলাদেশের বিজয় দিবস কত তারিখে?", "১৬ ডিসেম্বর"),
            ("রবীন্দ্রনাথ ঠাকুরের প্রথম মুদ্রিত কাব্যগ্রন্থ কোনটি?", "কবিকাহিনী"),
            ("রবীন্দ্রনাথ ঠাকুর কত সালে মৃত্যুবরণ করেন?", "১৯৪১ সালে"),
        ]
        for question, answer in history_cases:
            self.assertEqual(predict_verified(answer, by_question[question]), 1)

        cultural_cases = [
            ("সুন্দরবন বাংলাদেশের কোন কোন জেলায় অবস্থিত?", "খুলনা, সাতক্ষীরা ও বাগেরহাট জেলায়"),
            ("অধিকাংশ আধুনিক চাকমা জনগণ কোন ধরনের বৌদ্ধ ধর্ম পালন করে?", "থেরবাদ বৌদ্ধ ধর্ম"),
        ]
        for question, answer in cultural_cases:
            self.assertEqual(predict_verified(answer, by_question[question]), 1)

        forensic_cases = [
            ("‘পদাবলী’র প্রথম কবি কে?", "বিদ্যাপতি"),
            ("এখানে 'উড়ো চিঠি' বাগধারা/প্রবাদটির অর্থ কী?", "বেনামী বা অজ্ঞাতনামা চিঠি"),
            ("এখানে 'ফপর দালালি' বাগধারা/প্রবাদটির অর্থ কী?", "অনধিকার চর্চা"),
            ("এখানে 'রাঘব বোয়াল' বাগধারা/প্রবাদটির অর্থ কী?", "প্রভাবশালী দুর্নীতিবাজ"),
            ("পদ্মা নদী কোথায় যমুনা নদীর সাথে মিলিত হয়?", "গোয়ালন্দে"),
            ("বাক্‌ + দান = বাগদান সন্ধিটি নিপাতনে সিদ্ধ?", "হ্যাঁ"),
        ]
        for question, answer in forensic_cases:
            self.assertEqual(predict_verified(answer, by_question[question]), 1)

        makal = by_question["এখানে 'মাকাল ফল' প্রবাদ/বাগধারাটির অর্থ কী?"]
        self.assertEqual(predict_verified("অন্তঃসারশূন্য", makal), 1)
        self.assertEqual(predict_verified("ভেতরে-বাইরে সমান গুণসম্পন্ন বস্তু", makal), 0)

        image_cases = [
            (
                "ভারসাম্য রক্ষা করতে নির্দেশিত স্থানে কত কেজি ওজন রাখতে হবে?![](https://web.livemcq.com/job-solution/bcs-question-bank/40th-bcs-question-solution-pdf/)",
                "১৪০",
            ),
            (
                "নিচের ক, খ, গ ও ঘ এই ৪টি বিকল্প নকশার মধ্যে চিত্রের প্রশ্নবােধক চিহ্নের ঘরে কোন নকশাটি বসবে?![](https://web.livemcq.com/job-solution/bcs-question-bank/43-bcs-question-solution-pdf/)",
                "ক",
            ),
        ]
        for question, answer in image_cases:
            self.assertEqual(predict_verified(answer, by_question[question]), 1)

        talkana = by_question["এখানে 'তালকানা' বাগধারা/প্রবাদটির অর্থ কী?"]
        self.assertEqual(predict_verified("ছন্দ বা তাল বুঝতে না পারা, বেসুরা", talkana), 1)

    def test_verified_idiom_meanings(self) -> None:
        by_question = {item["question"]: item for item in self.entries.values()}
        cases = [
            ("ঊনপাঁজুরে", "হতভাগ্য", "চালাক ও ধুরন্ধর ব্যক্তি"),
            ("গোঁফ খেজুরে", "নিতান্ত অলস", "রসিক ও আমুদে ব্যক্তি"),
            ("ঢাকের কাঠি", "তোষামোদকারী", "প্রহরী"),
            ("তাসের ঘর", "ক্ষণস্থায়ী বা ভঙ্গুর জিনিস", "স্থায়ী অবস্থা"),
            ("রাবণের চিতা", "চির অশান্তি", "সাময়িক বিপদ"),
            ("লেফাফা দুরস্ত", "পরিপাটি", "সব দিক থেকেই সুন্দর ও যোগ্য"),
            ("শিরে সংক্রান্তি", "আসন্ন বিপদ", "সদ্য বিপদমুক্তি"),
            ("ব্যাঙের আধুলি", "অতি সামান্য ধন", "প্রকৃত ধনসম্পদের অধিকারী হওয়া"),
            ("শাঁখের করাত", "উভয় সংকট, দুদিক থেকেই বিপদ", "একমুখী সুবিধা"),
            ("হাতের পাঁচ", "শেষ সম্বল", "সবচেয়ে অপ্রয়োজনীয় বস্তু"),
            ("কূপমণ্ডূক", "সীমিত জ্ঞানসম্পন্ন, সংকীর্ণমনা ব্যক্তি", "উদার বিশ্বদৃষ্টির ব্যক্তি"),
            ("অহি-নকুল সম্বন্ধ", "ভীষণ শত্রুতা", "গভীর বন্ধুত্ব"),
        ]
        for idiom, correct, wrong in cases:
            question = f"এখানে '{idiom}' প্রবাদ/বাগধারাটির অর্থ কী?"
            self.assertEqual(predict_verified(correct, by_question[question]), 1)
            self.assertEqual(predict_verified(wrong, by_question[question]), 0)


if __name__ == "__main__":
    unittest.main()
