"""
Indexers for loading videos and images.

## Usage on DVDs

You can use the [dvdtools](https://github.com/Jaded-Encoding-Thaumaturgy/vs-preview-plugins/tree/master/dvdtools)
vs-preview plugin to determine chapter splits.

### Basic Usage

Previewing a title and dumping AC3 audio:

```py
from vssource import IsoFile
from vstools import set_output

# Create an IsoFile object from a DVD ISO or folder path
# This will automatically detect and use the best available indexer
iso = IsoFile("./DVD_VIDEOS/Suzumiya_2009_DVD/KABA_6001.ISO")

# Get a Title object representing the first title on the DVD
# Titles are 1-indexed
title = iso.get_title(1)

# Print information about the title
print(title)

# Preview the video in your previewer
# This outputs the entire video track of the title
title.preview()

# Extract the AC3 audio from the first audio track (index 0)
# This dumps the raw AC3 stream to a file
title.dump_ac3("full_title.ac3", 0)
```

Splitting titles:

```py
# Split a title into multiple parts at specific chapter boundaries
# This splits at chapters 6 and 11, creating 3 parts:
# - ep1: chapters 1-5
# - ep2: chapters 6-10
# - ep3: chapters 11-end
ep1, ep2, ep3 = title.split_at([6, 11])

# Split a title into specific chapter ranges
# Each tuple defines (start_chapter, end_chapter) inclusive
# This creates 3 parts from chapters 1-5, 6-10, and 11-15
# Any chapters after 15 are dropped
ep1, ep2, ep3 = title.split_ranges([(1, 5), (6, 10), (11, 15)])

# Split individual ranges one at a time
# Using -1 as end_chapter takes all remaining chapters
ep1 = title.split_range(1, 5)  # Chapters 1-5
ep2 = title.split_range(6, 10)  # Chapters 6-10
ep3 = title.split_range(11, -1)  # Chapter 11 to end

# Preview the full title and its splits in your video previewer
# This will output the full title and the individual parts after splitting at chapters 6 and 11
title.preview(title.split_at([6, 11]))

# Dump the first episode split's AC3 audio to a file and get the audio offset.
# The returned value is the offset in seconds between the start of the audio
# and the start of the video. This is useful for syncing audio and video,
# since DVD AC3 audio frames don't perfectly align with chapter boundaries.
# A positive value means there is extra audio at the start that needs trimming.
ep1_ac3_offset = ep1.ac3("ep1.ac3", 0)
```

### Advanced Usage

Trimming unwanted frames from a title

```py
# Sometimes DVDs have junk frames at the end of titles that we want to remove
title1 = iso.get_title(1)
# Remove 609 frames from the last chapter
title1.chapters[-1] -= 609
# Split into episodes at chapters 7 and 12, and preview the splits
title1.preview(title1.split_at([7, 12]))
```

Batch processing multiple titles

```py
# Here we process titles 2-5 which contain episodes
# Each title has some junk frames at the start we want to trim
splits = []

for title_num in range(2, 6):
    title = iso.get_title(title_num)
    # Add 180 frames offset to first chapter to skip junk frames
    title.chapters[0] += 180
    # Split and store the processed title
    splits.append(title.split_at([]))
    # Preview to verify the trim looks correct
    title.preview(splits[-1])

# Extract audio from first split and get sync offset
audio_offset = splits[0].ac3("ep1.ac3")
print(f"Audio offset: {audio_offset:.3f} seconds")
```

Working with multi-angle content and different audio tracks

```py
# Get title 4 with angle 1 (Japanese video) and audio track 1
# rff_mode=2 enables repeat-field flags for proper frame timing
japanese = iso.get_title(4, angle_nr=1, rff_mode=2).split_at([5, 10, 15], audio=1)

# Preview episode 1 Japanese video and audio
japanese[0].preview()

# Get same title with angle 2 (Italian video) and audio track 0
italian = iso.get_title(4, angle_nr=2, rff_mode=2).split_at([5, 10, 15], audio=0)

# Preview episode 2 Italian video and audio
italian[1].preview()
```

The `Title` class provides two main methods for splitting DVD titles into segments:

`split_at([chapters])`: Splits a title at the specified chapter numbers, similar
to how mkvmerge handles chapter splits. The splits occur before each specified
chapter. For example:

- `split_at([5])` splits the title into two parts:
  1. Chapters 1-4
  2. Chapters 5-end

`split_range(start, end)`: Extracts a range of chapters inclusively. For example:

- `split_range(2, 4)` extracts chapters 2-4

The output chapters are 1-based and match the DVD chapter numbers. This matches how DVD chapters work, where:

- Chapter 1 is the first chapter
- Splits occur at chapter boundaries
- Chapter numbers match what you see in DVD menus and players

```py
+---+----------+------+---+---+--------+--------+---+
| 1 |     2    |   3  | 4 | 5 |    6   |    7   | 8 |
+---+----------+------+---+---+--------+--------+---+

split_at([5])
+---+----------+------+---+
| 1 |     2    |   3  | 4 |  # First segment: Chapters 1-4
+---+----------+------+---+
                           +---+--------+--------+---+
                           | 5 |    6   |    7   | 8 |  # Second segment: Chapters 5-8
                           +---+--------+--------+---+

split_at([3, 6])
+---+----------+
| 1 |     2    |  # First segment: Chapters 1-2
+---+----------+
               +------+---+---+
               |   3  | 4 | 5 |  # Second segment: Chapters 3-5
               +------+---+---+
                              +--------+--------+---+
                              |    6   |    7   | 8 |  # Third segment: Chapters 6-8
                              +--------+--------+---+

split_range(2, 4)
    +----------+------+---+
    |     2    |   3  | 4 |  # Extracts just chapters 2-4
    +----------+------+---+
```
"""

from .formats import *
from .indexers import *
