"""Independent examples for quotation-bound duration values."""
import unittest

from rfp_rag.generation.duration_values import missing_duration_values


def claim(text,*quotes,category='duration'):
    return {'category':category,'text':text,'citations':[{'quote':q} for q in quotes]}


class DurationValueTests(unittest.TestCase):
    def test_all_claimed_periods_need_a_quoted_value(self):
        value=claim('추정에는 3개월과 9개월이 반영되었다.','안정화 이후 9개월','총 12개월')
        self.assertEqual(missing_duration_values(value),['3개월'])

    def test_multiple_independent_quotations_can_support_multiple_values(self):
        value=claim('구축 4개월과 운영 11개월을 구분한다.','구축: 4개월','운영: 11개월')
        self.assertEqual(missing_duration_values(value),[])

    def test_substring_value_and_unquoted_chunk_numbers_are_not_support(self):
        value=claim('기간은 3개월이다.','기간은 13개월이다.')
        value['citations'][0]['chunk_text']='전체 청크 어딘가에는 3개월이 있다.'
        self.assertEqual(missing_duration_values(value),['3개월'])

    def test_whitespace_and_thousands_grouping_are_the_only_normalizations(self):
        self.assertEqual(missing_duration_values(claim('1,200일 및 2.5개월','1 200 일','2.5 개 월')),[])
        self.assertEqual(missing_duration_values(claim('3개월','3.0개월')),['3개월'])
        self.assertEqual(missing_duration_values(claim('3,5개월','35개월')),['3,5개월'])

    def test_units_are_not_converted(self):
        self.assertEqual(missing_duration_values(claim('12개월','1년','365일')),['12개월'])
        self.assertEqual(missing_duration_values(claim('1달','1개월')),['1달'])

    def test_individual_quote_boundaries_are_preserved(self):
        self.assertEqual(missing_duration_values(claim('기간 7개월','7','개월')),['7개월'])

    def test_duplicates_are_bounded_and_other_categories_are_unchanged(self):
        self.assertEqual(missing_duration_values(claim('7개월, 7개월 및 3주')),['7개월','3주'])
        self.assertEqual(missing_duration_values(claim('예산 7개월분',category='budget')),[])
        self.assertEqual(missing_duration_values(claim('계약일부터 정해진 종료일까지')),[])


if __name__=='__main__':unittest.main()
