# Results directory

This directory intentionally contains no checkpoints, renders, or copied
datasets. Complete runners write one JSON summary per scene into the chosen
model directory. A result is valid evidence only when its protocol records
zero future-frame use before evaluation and its final checkpoint passes the
cold-reload equality check.

