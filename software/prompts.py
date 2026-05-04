"""
Zero-shot CLIP prompt templates for all 40 CelebA attributes.

Structure: PROMPTS[attr_name]["pos" | "neg"] -> list of text descriptions.
Attribute names match the column names in data/metadata.csv.

Usage (ensemble, standard CLIP zero-shot):
    import clip
    import torch
    from software.prompts import PROMPTS

    def encode_prompts(model, texts, device):
        tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens)
        return embs.mean(dim=0, keepdim=True)  # average over variants

    pos_emb = encode_prompts(model, PROMPTS["eyeglasses"]["pos"], device)
    neg_emb = encode_prompts(model, PROMPTS["eyeglasses"]["neg"], device)
"""

PROMPTS: dict[str, dict[str, list[str]]] = {

    "5_o_clock_shadow": {
        "pos": [
            "a photo of a person with a five o'clock shadow",
            "a photo of a person with light stubble on their face",
            "a photo of a man with short facial stubble",
            "a photo of a person with a day's worth of beard growth",
        ],
        "neg": [
            "a photo of a person with a clean-shaven face",
            "a photo of a person with no facial stubble",
            "a photo of a person with smooth skin and no beard growth",
        ],
    },

    "arched_eyebrows": {
        "pos": [
            "a photo of a person with arched eyebrows",
            "a photo of a person with highly curved eyebrows",
            "a photo of a person with elegantly arched brows",
        ],
        "neg": [
            "a photo of a person with straight eyebrows",
            "a photo of a person with flat, unarched eyebrows",
            "a photo of a person with horizontal brows",
        ],
    },

    "attractive": {
        "pos": [
            "a photo of an attractive person",
            "a photo of a good-looking person",
            "a photo of a beautiful person",
            "a photo of a handsome or pretty person",
        ],
        "neg": [
            "a photo of a plain-looking person",
            "a photo of an average-looking person",
            "a photo of an unattractive person",
        ],
    },

    "bags_under_eyes": {
        "pos": [
            "a photo of a person with bags under their eyes",
            "a photo of a person with puffy under-eye areas",
            "a photo of a person with dark circles and swelling under the eyes",
            "a photo of a tired-looking person with under-eye bags",
        ],
        "neg": [
            "a photo of a person with no bags under their eyes",
            "a photo of a person with smooth skin under the eyes",
            "a photo of a person with fresh, rested-looking eyes",
        ],
    },

    "bald": {
        "pos": [
            "a photo of a bald person",
            "a photo of a person with no hair on their head",
            "a photo of a person with a shaved head",
            "a photo of a person who is completely bald",
        ],
        "neg": [
            "a photo of a person with hair on their head",
            "a photo of a person with a full head of hair",
            "a photo of a person who is not bald",
        ],
    },

    "bangs": {
        "pos": [
            "a photo of a person with bangs",
            "a photo of a person with hair covering their forehead",
            "a photo of a person with a fringe hairstyle",
            "a photo of a person with bangs across their forehead",
        ],
        "neg": [
            "a photo of a person without bangs",
            "a photo of a person with their forehead fully visible",
            "a photo of a person with hair swept back from the forehead",
        ],
    },

    "big_lips": {
        "pos": [
            "a photo of a person with big lips",
            "a photo of a person with full, prominent lips",
            "a photo of a person with large, voluminous lips",
        ],
        "neg": [
            "a photo of a person with thin lips",
            "a photo of a person with small lips",
            "a photo of a person with narrow lips",
        ],
    },

    "big_nose": {
        "pos": [
            "a photo of a person with a big nose",
            "a photo of a person with a large, prominent nose",
            "a photo of a person with a wide nose",
        ],
        "neg": [
            "a photo of a person with a small nose",
            "a photo of a person with a petite nose",
            "a photo of a person with a narrow, delicate nose",
        ],
    },

    "black_hair": {
        "pos": [
            "a photo of a person with black hair",
            "a photo of a person with dark black hair",
            "a photo of a person with jet-black hair",
        ],
        "neg": [
            "a photo of a person without black hair",
            "a photo of a person with non-black hair",
            "a photo of a person with blonde, brown, or gray hair",
        ],
    },

    "blond_hair": {
        "pos": [
            "a photo of a person with blond hair",
            "a photo of a person with blonde hair",
            "a photo of a person with light golden hair",
            "a photo of a person with fair-colored hair",
        ],
        "neg": [
            "a photo of a person without blond hair",
            "a photo of a person with dark or brown hair",
            "a photo of a person with non-blonde hair",
        ],
    },

    "blurry": {
        "pos": [
            "a blurry photo of a person",
            "an out-of-focus photo of a person",
            "a low-quality blurry image of a person",
        ],
        "neg": [
            "a sharp, clear photo of a person",
            "a high-quality, in-focus photo of a person",
            "a crisp and well-focused photo of a person",
        ],
    },

    "brown_hair": {
        "pos": [
            "a photo of a person with brown hair",
            "a photo of a person with dark brown hair",
            "a photo of a person with chestnut-colored hair",
        ],
        "neg": [
            "a photo of a person without brown hair",
            "a photo of a person with blonde, black, or gray hair",
            "a photo of a person with non-brown hair",
        ],
    },

    "bushy_eyebrows": {
        "pos": [
            "a photo of a person with bushy eyebrows",
            "a photo of a person with thick, heavy eyebrows",
            "a photo of a person with dense, prominent brows",
        ],
        "neg": [
            "a photo of a person with thin eyebrows",
            "a photo of a person with light, sparse eyebrows",
            "a photo of a person with narrow, delicate brows",
        ],
    },

    "chubby": {
        "pos": [
            "a photo of a chubby person",
            "a photo of a person with a round, full face",
            "a photo of an overweight person",
            "a photo of a heavyset person with chubby cheeks",
        ],
        "neg": [
            "a photo of a slim person",
            "a photo of a person with a lean face",
            "a photo of a person with a thin face and defined cheekbones",
        ],
    },

    "double_chin": {
        "pos": [
            "a photo of a person with a double chin",
            "a photo of a person with extra flesh under the chin",
            "a photo of a person with a prominent double chin",
        ],
        "neg": [
            "a photo of a person without a double chin",
            "a photo of a person with a well-defined jawline",
            "a photo of a person with no extra chin",
        ],
    },

    "eyeglasses": {
        "pos": [
            "a photo of a person wearing eyeglasses",
            "a photo of a person with glasses",
            "a photo of a person wearing spectacles",
            "a photo of a person in glasses",
        ],
        "neg": [
            "a photo of a person not wearing eyeglasses",
            "a photo of a person without glasses",
            "a photo of a person with no spectacles",
        ],
    },

    "goatee": {
        "pos": [
            "a photo of a person with a goatee",
            "a photo of a man with a goatee beard",
            "a photo of a person with a small beard on the chin",
        ],
        "neg": [
            "a photo of a person without a goatee",
            "a photo of a person with no chin beard",
            "a photo of a clean-shaven person with no goatee",
        ],
    },

    "gray_hair": {
        "pos": [
            "a photo of a person with gray hair",
            "a photo of a person with grey hair",
            "a photo of a person with silver hair",
            "a photo of an older person with gray hair",
        ],
        "neg": [
            "a photo of a person without gray hair",
            "a photo of a person with naturally colored hair",
            "a photo of a person with dark or blonde hair, not gray",
        ],
    },

    "heavy_makeup": {
        "pos": [
            "a photo of a person wearing heavy makeup",
            "a photo of a person with a lot of makeup on",
            "a photo of a person with dramatic, full-face makeup",
            "a photo of a person wearing bold and heavy cosmetics",
        ],
        "neg": [
            "a photo of a person wearing no makeup",
            "a photo of a person with a natural, bare face",
            "a photo of a person wearing minimal or no makeup",
        ],
    },

    "high_cheekbones": {
        "pos": [
            "a photo of a person with high cheekbones",
            "a photo of a person with prominent, elevated cheekbones",
            "a photo of a person with defined, high facial bone structure",
        ],
        "neg": [
            "a photo of a person with low or flat cheekbones",
            "a photo of a person with a round face and no prominent cheekbones",
            "a photo of a person without high cheekbones",
        ],
    },

    "male": {
        "pos": [
            "a photo of a man",
            "a photo of a male person",
            "a photo of a guy",
        ],
        "neg": [
            "a photo of a woman",
            "a photo of a female person",
            "a photo of a lady",
        ],
    },

    "mouth_slightly_open": {
        "pos": [
            "a photo of a person with their mouth slightly open",
            "a photo of a person with lips parted",
            "a photo of a person with a slightly open mouth",
        ],
        "neg": [
            "a photo of a person with their mouth closed",
            "a photo of a person with lips pressed together",
            "a photo of a person with a closed mouth",
        ],
    },

    "mustache": {
        "pos": [
            "a photo of a person with a mustache",
            "a photo of a man with a mustache above the lip",
            "a photo of a person with facial hair on their upper lip",
        ],
        "neg": [
            "a photo of a person without a mustache",
            "a photo of a person with a clean upper lip",
            "a photo of a person with no mustache",
        ],
    },

    "narrow_eyes": {
        "pos": [
            "a photo of a person with narrow eyes",
            "a photo of a person with small, squinting eyes",
            "a photo of a person with slender, narrow-set eyes",
        ],
        "neg": [
            "a photo of a person with wide, open eyes",
            "a photo of a person with large, round eyes",
            "a photo of a person with wide-open eyes",
        ],
    },

    "no_beard": {
        "pos": [
            "a photo of a person with no beard",
            "a photo of a clean-shaven person",
            "a photo of a person with no facial hair",
            "a photo of a person with a smooth, beardless face",
        ],
        "neg": [
            "a photo of a person with a beard",
            "a photo of a person with facial hair",
            "a photo of a bearded person",
        ],
    },

    "oval_face": {
        "pos": [
            "a photo of a person with an oval face shape",
            "a photo of a person with an elongated, oval-shaped face",
            "a photo of a person with a classic oval face",
        ],
        "neg": [
            "a photo of a person with a round or square face shape",
            "a photo of a person with a non-oval face",
            "a photo of a person with a wide or angular face",
        ],
    },

    "pale_skin": {
        "pos": [
            "a photo of a person with pale skin",
            "a photo of a person with fair, light complexion",
            "a photo of a person with very light skin tone",
        ],
        "neg": [
            "a photo of a person with dark or tanned skin",
            "a photo of a person with a warm or deep skin tone",
            "a photo of a person with non-pale skin",
        ],
    },

    "pointy_nose": {
        "pos": [
            "a photo of a person with a pointy nose",
            "a photo of a person with a sharp, pointed nose",
            "a photo of a person with a narrow, pointed nose tip",
        ],
        "neg": [
            "a photo of a person with a round or flat nose",
            "a photo of a person with a broad, non-pointy nose",
            "a photo of a person with a blunt nose tip",
        ],
    },

    "receding_hairline": {
        "pos": [
            "a photo of a person with a receding hairline",
            "a photo of a person whose hair is thinning at the temples",
            "a photo of a person with a high, receding hairline",
        ],
        "neg": [
            "a photo of a person with a full, non-receding hairline",
            "a photo of a person with thick hair at the forehead",
            "a photo of a person with no receding hairline",
        ],
    },

    "rosy_cheeks": {
        "pos": [
            "a photo of a person with rosy cheeks",
            "a photo of a person with pink, flushed cheeks",
            "a photo of a person with a rosy, healthy complexion on the cheeks",
        ],
        "neg": [
            "a photo of a person with pale or neutral cheeks",
            "a photo of a person with no rosy color on the cheeks",
            "a photo of a person with a non-flushed complexion",
        ],
    },

    "sideburns": {
        "pos": [
            "a photo of a person with sideburns",
            "a photo of a man with sideburns along the face",
            "a photo of a person with hair growing on the sides of the face",
        ],
        "neg": [
            "a photo of a person without sideburns",
            "a photo of a person with clean-shaved sides of the face",
            "a photo of a person with no sideburns",
        ],
    },

    "smiling": {
        "pos": [
            "a photo of a smiling person",
            "a photo of a person with a smile on their face",
            "a photo of a happy, smiling person",
            "a photo of a person grinning",
        ],
        "neg": [
            "a photo of a person with a neutral expression",
            "a photo of a person who is not smiling",
            "a photo of a serious-looking person",
        ],
    },

    "straight_hair": {
        "pos": [
            "a photo of a person with straight hair",
            "a photo of a person with sleek, straight hair",
            "a photo of a person with flat, non-curly hair",
        ],
        "neg": [
            "a photo of a person with curly or wavy hair",
            "a photo of a person with non-straight hair",
            "a photo of a person with wavy or curled hair",
        ],
    },

    "wavy_hair": {
        "pos": [
            "a photo of a person with wavy hair",
            "a photo of a person with gently curled, wavy hair",
            "a photo of a person with flowing, wavy hair",
        ],
        "neg": [
            "a photo of a person with straight or tightly curled hair",
            "a photo of a person with non-wavy hair",
            "a photo of a person with flat, straight hair",
        ],
    },

    "wearing_earrings": {
        "pos": [
            "a photo of a person wearing earrings",
            "a photo of a person with earrings in their ears",
            "a photo of a person accessorized with earrings",
        ],
        "neg": [
            "a photo of a person not wearing earrings",
            "a photo of a person with no earrings",
            "a photo of a person with bare ears",
        ],
    },

    "wearing_hat": {
        "pos": [
            "a photo of a person wearing a hat",
            "a photo of a person with a hat on their head",
            "a photo of a person in a hat",
        ],
        "neg": [
            "a photo of a person not wearing a hat",
            "a photo of a person with no hat",
            "a photo of a person with their bare head visible",
        ],
    },

    "wearing_lipstick": {
        "pos": [
            "a photo of a person wearing lipstick",
            "a photo of a person with lipstick on their lips",
            "a photo of a person with colored lips from lipstick",
        ],
        "neg": [
            "a photo of a person not wearing lipstick",
            "a photo of a person with natural, bare lips",
            "a photo of a person with no lipstick",
        ],
    },

    "wearing_necklace": {
        "pos": [
            "a photo of a person wearing a necklace",
            "a photo of a person with a necklace around their neck",
            "a photo of a person accessorized with a necklace",
        ],
        "neg": [
            "a photo of a person not wearing a necklace",
            "a photo of a person with a bare neck and no necklace",
            "a photo of a person with no necklace",
        ],
    },

    "wearing_necktie": {
        "pos": [
            "a photo of a person wearing a necktie",
            "a photo of a person with a tie around their neck",
            "a photo of a formally dressed person wearing a necktie",
        ],
        "neg": [
            "a photo of a person not wearing a necktie",
            "a photo of a person with no tie",
            "a photo of a casually dressed person without a necktie",
        ],
    },

    "young": {
        "pos": [
            "a photo of a young person",
            "a photo of a youthful-looking person",
            "a photo of a person who looks young",
            "a photo of a person in their twenties or thirties",
        ],
        "neg": [
            "a photo of an older person",
            "a photo of a middle-aged or elderly person",
            "a photo of a person who looks old",
        ],
    },

}

assert len(PROMPTS) == 40, f"Expected 40 attributes, got {len(PROMPTS)}"
