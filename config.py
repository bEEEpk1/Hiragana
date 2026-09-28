from __future__ import annotations

# 46 базовых символов хираганы в стандартном порядке gojuon.
# aliases учитывают структуру скачанного датасета, где гласные названы aa/ii/uu/ee/oo.
HIRAGANA_CLASSES = [
    ("a", "あ", ("a", "aa", "あ")),
    ("i", "い", ("i", "ii", "い")),
    ("u", "う", ("u", "uu", "う")),
    ("e", "え", ("e", "ee", "え")),
    ("o", "お", ("o", "oo", "お")),
    ("ka", "か", ("ka", "か")),
    ("ki", "き", ("ki", "き")),
    ("ku", "く", ("ku", "く")),
    ("ke", "け", ("ke", "け")),
    ("ko", "こ", ("ko", "こ")),
    ("sa", "さ", ("sa", "さ")),
    ("shi", "し", ("shi", "si", "し")),
    ("su", "す", ("su", "す")),
    ("se", "せ", ("se", "せ")),
    ("so", "そ", ("so", "そ")),
    ("ta", "た", ("ta", "た")),
    ("chi", "ち", ("chi", "ti", "ち")),
    ("tsu", "つ", ("tsu", "tu", "つ")),
    ("te", "て", ("te", "て")),
    ("to", "と", ("to", "と")),
    ("na", "な", ("na", "な")),
    ("ni", "に", ("ni", "に")),
    ("nu", "ぬ", ("nu", "ぬ")),
    ("ne", "ね", ("ne", "ね")),
    ("no", "の", ("no", "の")),
    ("ha", "は", ("ha", "は")),
    ("hi", "ひ", ("hi", "ひ")),
    ("fu", "ふ", ("fu", "hu", "ふ")),
    ("he", "へ", ("he", "へ")),
    ("ho", "ほ", ("ho", "ほ")),
    ("ma", "ま", ("ma", "ま")),
    ("mi", "み", ("mi", "み")),
    ("mu", "む", ("mu", "む")),
    ("me", "め", ("me", "め")),
    ("mo", "も", ("mo", "も")),
    ("ya", "や", ("ya", "や")),
    ("yu", "ゆ", ("yu", "ゆ")),
    ("yo", "よ", ("yo", "よ")),
    ("ra", "ら", ("ra", "ら")),
    ("ri", "り", ("ri", "り")),
    ("ru", "る", ("ru", "る")),
    ("re", "れ", ("re", "れ")),
    ("ro", "ろ", ("ro", "ろ")),
    ("wa", "わ", ("wa", "わ")),
    ("wo", "を", ("wo", "を")),
    ("nn", "ん", ("nn", "n", "ん")),
]

NUM_CLASSES = len(HIRAGANA_CLASSES)
ROMAJI = [item[0] for item in HIRAGANA_CLASSES]
KANA = [item[1] for item in HIRAGANA_CLASSES]

ALIAS_TO_CLASS_ID: dict[str, int] = {}
for class_id, (_, _, aliases) in enumerate(HIRAGANA_CLASSES):
    for alias in aliases:
        ALIAS_TO_CLASS_ID[alias.lower()] = class_id

IMAGE_SIZE = 32
CONTENT_SIZE = 26
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
