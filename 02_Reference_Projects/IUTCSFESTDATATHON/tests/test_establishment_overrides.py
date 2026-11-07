import unittest

from establishment_overrides import decide


class EstablishmentTests(unittest.TestCase):
    def test_establishment_vs_status(self):
        context = "বাংলাদেশ জাতীয় জাদুঘর ২০ মার্চ ১৯১৩ সালে প্রতিষ্ঠিত হয়; ১৯৮৩ সালে জাতীয় জাদুঘরের মর্যাদা পায়।"
        self.assertEqual(decide("বাংলাদেশ জাতীয় জাদুঘর কত সালে প্রতিষ্ঠিত হয়?", "১৯৮৩", context)[0], 0)
        self.assertEqual(decide("কত সালে বাংলাদেশ জাতীয় জাদুঘরকে জাতীয় জাদুঘরের মর্যাদা দেওয়া হয়?", "১৯৮৩", context)[0], 1)

    def test_parenthetical_parent_date_removed(self):
        context = "কুয়েতি প্রগতিশীল আন্দোলন একটি দল। দলটি ২০১১ সালে পিপলস ইউনিয়নের (১৯৭৫ সালে প্রতিষ্ঠিত) সম্প্রসারণ হিসাবে প্রতিষ্ঠিত হয়।"
        self.assertEqual(decide("কুয়েতি প্রগতিশীল আন্দোলন কত সালে প্রতিষ্ঠিত হয়?", "২০১১", context)[0], 1)

    def test_registration_is_not_establishment(self):
        context = "কোয়ার্টেরা একটি সংগঠন। এই সমিতিটি ২০১১ সালে নিবন্ধিত হয়। এই সমিতিটি প্রথম ২০০৯ সালে প্রতিষ্ঠিত হয়েছিল।"
        self.assertEqual(decide("কোয়ার্টেরা কত সালে প্রতিষ্ঠিত হয়?", "২০১১", context)[0], 0)

    def test_ambiguous_same_name_abstains(self):
        context = "শ্রীপুর সরকারি কলেজ, গাজীপুর ১৯৬৮ সালে প্রতিষ্ঠিত। শ্রীপুর সরকারি কলেজ, মাগুরা ১৯৬৯ সালে প্রতিষ্ঠিত।"
        self.assertIsNone(decide("শ্রীপুর সরকারি কলেজ কত সালে প্রতিষ্ঠিত হয়?", "১৯৬৯", context)[0])

    def test_construction_start_vs_completion(self):
        context = "কান্তনগর মন্দিরের নির্মাণ ১৭০৪ সালে শুরু হয় এবং ১৭৫২ সালে সম্পন্ন হয়।"
        self.assertEqual(decide("কান্তনগর মন্দিরের নির্মাণ কাজ কত সালে শুরু হয়?", "১৭৫২", context)[0], 0)

    def test_nearby_quantity_is_not_a_year(self):
        context = "রাজশাহী ক্যাডেট কলেজ ১১০ একর জমির ওপর প্রতিষ্ঠিত। এটি ১১ ফেব্রুয়ারি ১৯৬৬ সালে উদ্বোধন করা হয়।"
        self.assertIsNone(decide("রাজশাহী ক্যাডেট কলেজ কত সালে প্রতিষ্ঠিত হয়?", "১৯৬৬", context)[0])

    def test_acronym_links_history_to_subject(self):
        context = "লেবার রিসার্চ ডিপার্টমেন্ট (এলআরডি) একটি গবেষণা সংস্থা। এলআরডি এর সূচনা হয়েছিল ১৯১২ সালে। এর বুলেটিন ১৯১৭ সালে প্রতিষ্ঠিত হয়েছিল।"
        self.assertEqual(decide("লেবার রিসার্চ ডিপার্টমেন্ট কত সালে প্রতিষ্ঠিত হয়?", "১৯১৭", context)[0], 0)

    def test_historical_shorter_name_is_same_entity(self):
        context = "বিয়াম ল্যাবরেটরি স্কুল এন্ড কলেজ একটি প্রতিষ্ঠান। ২০০৭ সালে বিয়াম ল্যাবরেটরি স্কুল প্রতিষ্ঠিত হয়। ২০১৩ সালে প্রতিষ্ঠানটির নাম হয় বিয়াম ল্যাবরেটরি স্কুল এন্ড কলেজ।"
        self.assertEqual(decide("বিয়াম ল্যাবরেটরি স্কুল এন্ড কলেজ কত সালে প্রতিষ্ঠিত হয়?", "২০১৩", context)[0], 0)

    def test_related_subentity_does_not_make_year_ambiguous(self):
        context = "১৯৩৭ সালে বকশীগঞ্জ ইউনিয়ন পরিষদ গঠিত হয়। ১৯৮৩ সালে বকশীগঞ্জ উপজেলা গঠিত হয়।"
        self.assertEqual(decide("বকশীগঞ্জ উপজেলা কত সালে প্রতিষ্ঠিত হয়?", "১৯৩৭", context)[0], 0)

    def test_single_token_spelling_variant_links_pronoun(self):
        context = "কোয়ার্টেরা একটি সংগঠন। পরে এই সমিতিটি প্রথম ২০০৯ সালে প্রতিষ্ঠিত হয়েছিল।"
        self.assertEqual(decide("কোয়ার্টিরা কত সালে প্রতিষ্ঠিত হয়?", "২০১১", context)[0], 0)

    def test_person_is_not_an_established_organization(self):
        context = "বীরেন্দ্র কিশোর রোয়াজা ছিলেন একজন বাংলাদেশী সমাজকর্মী এবং রাজনীতিবিদ। ১৯৭২ সালে একটি সমিতি গঠিত হলে তিনি সভাপতি হন।"
        self.assertEqual(decide("বীরেন্দ্র কিশোর রোয়াজা কত সালে প্রতিষ্ঠিত হয়?", "১৯৭২", context)[0], 0)

    def test_proposal_date_is_not_establishment(self):
        context = "ফটিকছড়ি উত্তর একটি প্রস্তাবিত উপজেলা। ২০২৫ সালে এটি গঠনের প্রস্তাব গৃহীত হয়।"
        self.assertEqual(decide("ফটিকছড়ি উত্তর উপজেলা কত সালে প্রতিষ্ঠিত হয়?", "২০২৫", context)[0], 0)

    def test_parent_organization_year_is_not_local_branch_year(self):
        context = "মেট্রোপলিটন কমিউনিটি চার্চ অফ এডিনবার্গ ছিল একটি স্থানীয় মণ্ডলী। এমসিসি একটি আন্তর্জাতিক সম্প্রদায় যা ১৯৬৮ সালে প্রতিষ্ঠিত হয়।"
        self.assertEqual(decide("মেট্রোপলিটন কমিউনিটি চার্চ অফ এডিনবার্গ কত সালে প্রতিষ্ঠিত হয়?", "১৯৬৮", context)[0], 0)

    def test_mid_sentence_pronoun_preserves_target(self):
        context = "রানাপিং মাদ্রাসা একটি কওমি মাদ্রাসা। শিক্ষাপ্রতিষ্ঠান না থাকায় ১৯৩০ সালে এটি প্রতিষ্ঠা করেন একজন আলেম।"
        self.assertIsNone(decide("রানাপিং মাদ্রাসা কত সালে প্রতিষ্ঠিত হয়?", "১৯৩০", context)[0])


if __name__ == "__main__":
    unittest.main()
