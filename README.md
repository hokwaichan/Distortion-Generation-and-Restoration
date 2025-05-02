# Distortion Generation and Restoration with UV Maps

This project focuses on the generation of spatial distortions on images and videos, along with the creation of corresponding UV maps that represent pixel-wise displacements. It also supports the restoration of distorted frames using predicted UV maps.

## Project Goals

- Generate synthetic spatial distortions on visual data.
- Output UV maps that describe the pixel movement due to distortion.
- Apply predicted UV maps to restore distorted frames.
- Provide visualizations to support analysis and evaluation.

## Distortion Types

- **Random Field Distortion**: Produces smooth, noise-based deformations that simulate natural irregularities.
- **Radial Distortion**: Mimics optical distortions caused by camera lenses using adjustable parameters.
- **Vortex Distortion**: Applies rotational warping effects centered around the frame.
