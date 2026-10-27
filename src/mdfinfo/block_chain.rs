//! Cycle guard for the singly linked block chains of MDF3 and MDF4 files.
use log::warn;
use std::collections::HashSet;

/// Remembers the block offsets already walked in a linked list. MDF block chains
/// are acyclic, so a repeated offset means the file is corrupted and following
/// the link again would loop forever.
#[derive(Debug)]
pub(crate) struct BlockChain {
    /// block type, only used for the warning message
    kind: &'static str,
    visited: HashSet<i64>,
}

impl BlockChain {
    /// Starts tracking a chain of `kind` blocks ("FH", "DG", ...).
    pub(crate) fn new(kind: &'static str) -> Self {
        BlockChain {
            kind,
            visited: HashSet::new(),
        }
    }

    /// Records `offset` and returns false if it was already walked.
    pub(crate) fn visit(&mut self, offset: i64) -> bool {
        if self.visited.insert(offset) {
            true
        } else {
            warn!(
                "{} block cycle detected at 0x{offset:x}, stopping chain walk",
                self.kind
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visit_detects_repeat() {
        let mut chain = BlockChain::new("FH");
        assert!(chain.visit(168));
        assert!(chain.visit(224));
        assert!(!chain.visit(168));
        assert!(!chain.visit(224));
    }
}
