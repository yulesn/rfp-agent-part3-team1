"""Read-only compatibility for the exact, released ABCD bundle before web summaries.

Only the pinned web-summary integration is allowed to differ. Every original
retrieval, embedding, chunking, evidence, prompt and dependency binding remains
exact. Workspace manifests and paid vectors are never rewritten or rebuilt.
This verifier itself is trusted executable code (as is open_workspace).
"""

LEGACY_ABCD = {'__init__.py': 'b004168e50aa7686622509cf1f651a063fced1e6966b6a61cc8993c563e3aa8c',
 'adapters.py': '3db48caeb0b2a5829030bfb0547c1eb08c4cd6b43ed93c9f11ef9f2eb262ca6c',
 'cli.py': 'c8bd51b4c6e21d0fa1508d41234c7547edd00259b5ffba1ba3b6d14f95d3f5ca',
 'conversation/__init__.py': '8328e0e34d6a3eeb73b0b7f146857b4863e9de053623c87e0f93da32e32eb362',
 'conversation/focus.py': '4176ae786cebfc1273235f7d5cd4491e30e87b71c5c4b3112a4f609e1ea6f972',
 'conversation/literal_quotes.py': '7135562c0bdc5883faee2c29949f25a49ffeaf07867bb5741d28e1d931237062',
 'conversation/query.py': '13657cf1bb2421614d9a4d082ca8db58c560ce7b94d2716537d8000e738b6f93',
 'conversation/resolution.py': '0f83be7de85ddd4673284bea9fd4bde05e3c967c6663771a71d518a41e8465ad',
 'conversation/runtime.py': '5f155df59b1857697d460c0c446f8be076feb0058de487eff8e643c6eb387b29',
 'conversation/session.py': 'c649db96a00c6b405363a0f1ade9577b4ea5957e6861fb55b535d28ceac890a9',
 'defaults.json': 'c21be4255970ce1c1ebcd666aff1ccd25ca7a0702828bb72a77767ebf7133890',
 'evaluation/__init__.py': 'ae3778b4ee90f97f6274c23d262c81c32cd646631e56eacbd508d1b07859e063',
 'evaluation/diagnostics.py': '027b883a980740aded640f86cc8dece3574c7e3b4cc32fb0c94815aa01606350',
 'evaluation/evaluate.py': '585951565a9f1e53f2864f05ee046cd0bfc4258e3c8110a5779282a9a4b7d6e9',
 'evaluation/evidence.py': 'f7116bda4784fb4e3fc05b7758ffdee3e1e5ee591eda51394bec22d044a96a8e',
 'evaluation/review.py': '9abd02a335f22d75acab5ed06c8ae1f792288831cc81be3d7c39be348ae11139',
 'generation/__init__.py': 'd523769148a03f0648bd8c749344eacba311aca5d982ad822b51175a8f4706cd',
 'generation/citation_recovery.py': 'bd6827cd5525ba902f97d6ea8daa2ae6373a73c7c2ee5fe38cea62dc774826b2',
 'generation/comparison.py': '725e3414c58a57745ee85132ff4fa41010e9376e1bd2a0df405192da26c709c5',
 'generation/costs.py': '9b914ba06173b4cd0817dd561f4b46177cef18783efdf85b5e1746ea40ca7da6',
 'generation/duration_values.py': 'a0ba82464f8b28e8bb410d056d141867713fd97da067f00ec27a879993dbe581',
 'generation/evidence.py': 'c43fbd3dd15f53e3e86791f2a67e999ba0ba71e759e60ada147be3c22518ca4f',
 'generation/llm.py': 'dc21ed929a6c0161923a3326151e4d31f1ec0a519b15ad40de289ff9870ecf77',
 'generation/rag.py': '9be3ff3b25caa188457fef2dc595a9ab37a8cad79bccbede6c8a12594c9bf320',
 'generation/request_policy.py': '19f8b0d0aef8c57e4a573eaf708c4b8e9228f505393ed2d943b5b8c664f25e3c',
 'generation/runtime.py': 'b5cf9d1931f91db2d217960ae5bffd6647533a81aff8ccc412fbf149668a4967',
 'pilot/__init__.py': 'e91071c9ea45b6ff2975bd80e195ca49d9a9ebf0c6680040506b3eaf4b0e83eb',
 'pilot/claim_audit.py': '10eea6989eefbc502d94392accee28dfe52b283bb86d73a36770ca0e83fc7932',
 'pilot/pilot.html': '2c5751e858fd2ceafcb754ae4dee3d5f4c7cf5c58ae210a53185292cc53bf5db',
 'pilot/pilot.py': '77018a7f7468ff9a16420192b7ed3caf149cfff8c98712e6620fba9b7c513b7c',
 'pilot/validation.py': '3ace18629ff29b1aab3b1f38456b7e2e4c18c92ed617697e83e7234592d2de18',
 'pilot/web.py': 'b18227bfe4eb7116e1cb33eed6635db29603cb37875a9781a1bab6c1b8479697',
 'recovery/__init__.py': '8391504565094a906b24934d7fe8e4686403a420b17f25f5a979bb76e8ff4cbf',
 'recovery/client.py': '1bfeef276ba569f9ca2756d3d87cd172f0c0ccb5586d7351f81c9d763d55f572',
 'recovery/runner.py': '7bfdbc8831399bd2ec473e6bbb7bf85ffaab75632ec7a8eedad84d155b5ded2f',
 'retrieval/__init__.py': '23e7de457174bdb5ac8d9176f23817f0d5533f2e42d0aafe27c7966f2d538ff2',
 'retrieval/chunking.py': '9c9722857c3118982c8aa2950a1470929980425ba018510a12928c82af2fd066',
 'retrieval/evaluation.py': 'cd3ddcaeafbc3b094b9f308c5f79d1219cfe396de6175ced3ffc511814febc03',
 'retrieval/io.py': '7543b3b07af4b34ddfabdd87214849f66c2f2545f5966d03708e4190d3432b15',
 'retrieval/quality.py': '613aa009e297f0e9acadb26c53c79e64ab3fac69705a0f96243aaa610314c65e',
 'retrieval/retrieval.py': '7ae3fcad4ca551125a91e447d4478a6806eada764ff8ce188c04ac06eaccb240',
 'retrieval/semantic.py': '17c26552f6edf217a8ce98326c737e1c39f5c0e7b51db35922cfc692903fe938',
 'retrieval/session.py': '6f1ddd974e3cb19e248737aba37a4075bd55e5aeded16caa8eabcd6eaa5d4096',
 'summary_content/__init__.py': '99427af29c90c3a11fc52aa917198a321717730d7c4d1f83a56ff608c5536bb5',
 'summary_content/conditions.py': 'dceb415d3caa7e1a0327c4d400d6c38f73e957a8a8f3079e4a8623b75e55c55d',
 'summary_content/context.py': '71c4ab49a28bac4116a94e32a62a3baaa6c5deec063c9e6cc35a5fb913931462',
 'summary_content/pipeline.py': '03c46aec3dc318a986cd3c41c42288886aab136bd4e5181540257aca4e547cb5',
 'summary_content/storage.py': '86ad61b8c8112c4eed5eb0b1d8779cbf0bd7b67dc88810cbf5a097afc703bcfa',
 'summary_fields/__init__.py': '52c439a4b0ff2bdf409dd63734142e78a82256ece64d27af1c239f7811d56066',
 'summary_fields/fields.py': '56074b049e3935aa09ef6a04c80933bd6ed349fcdaf65259dbee3a80c966ceb9',
 'summary_full/__init__.py': 'b255bbb3949f5a0c430cac977c91a2373e0a980c92c8b82aeff45a28bbdaadfa',
 'summary_full/catalog_audit.py': '788c7365ab6f30228ac5a412fa88a0c4cb85e767e43e9b7503f9a6c5788f59f6',
 'summary_full/engine.py': 'a682126fb0fc21214e25846c7ce073e4e829fd3c3804dea1de80c3797f7cf67a',
 'summary_full/legacy.py': '21f1acc5cbe635487833d57729448aaf53704c9399f82bc1d6cf75d051cb1c98',
 'summary_full/review.py': 'e2a4bb7a3fee1757da8baaf69accf5069290b9bc637af6ce0f9ae2043fd2d1db',
 'summary_full/transport.py': 'f51ad82695524d1610506fe8d996360a6343b5909978be7c9b81a781b22e36cc',
 'workflows.py': 'd5dd7ad18942ed27e4015779469b6604b1d643ca197fa7c7096420796074504d',
 'workspace.py': 'e80a9a1b6cdbff161bc8426f1aa0087c5715692a379ce8b0f9f50f4de875394f'}

SUMMARY_INTEGRATION = {'pilot/pilot.html': '5f6ce9f7be450ad459d5707566949181d22f036278d906ad6de7bb239970b995',
 'pilot/summaries.py': '32ecd9164aa81cb4a2991e54e43bccfc79004aa5cd439b3f61b181a963945bfa',
 'pilot/web.py': 'f5e7cc848edcdb37a91ed7705c9c9f1dfca5db51dd6d135beefe0097a2fb39cc',
 'recovery/client.py': '04f130bb217641273b72c66e9bc09a62dcb00bcc54f6a765d99e9cd4d9e4979a',
 'workspace.py': 'e6d38088d9c9822e41605a0dd6d600fc542aaf2ca8e67991e342a369b2ef7feb'}

def compatible_package(saved, current):
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    if saved == current:
        return True
    if saved != LEGACY_ABCD:
        return False
    expected = {**LEGACY_ABCD, **SUMMARY_INTEGRATION}
    return (set(current) == set(expected) | {'workspace_compat.py'}
            and all(current.get(path) == sha for path, sha in expected.items()))
