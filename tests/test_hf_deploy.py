import unittest

from push_to_hf_space import DEFAULT_SPACE_ID, resolve_space_id


class HuggingFaceDeployTests(unittest.TestCase):
    def test_default_space_is_existing_project_space(self):
        self.assertEqual(DEFAULT_SPACE_ID, "rdlmoving/poser-outfit-changer")
        self.assertEqual(resolve_space_id("alice"), "rdlmoving/poser-outfit-changer")

    def test_bare_configured_name_uses_authenticated_username(self):
        self.assertEqual(resolve_space_id("alice", "my-space"), "alice/my-space")

    def test_explicit_owner_is_preserved(self):
        self.assertEqual(resolve_space_id("alice", "team/my-space"), "team/my-space")

    def test_default_does_not_require_username(self):
        self.assertEqual(resolve_space_id(""), "rdlmoving/poser-outfit-changer")

    def test_bare_override_requires_username(self):
        with self.assertRaises(ValueError):
            resolve_space_id("", "my-space")

    def test_malformed_space_id_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_space_id("alice", "one/two/three")


if __name__ == "__main__":
    unittest.main()
