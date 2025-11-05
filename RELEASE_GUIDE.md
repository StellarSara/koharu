# Creating GitHub Release 0.7.3

## Step 1: Commit and Push Changes

```powershell
# Stage all changes
git add .

# Commit with release message
git commit -m "release: 0.7.3 - Add PaddleOCR support for Chinese text"

# Push to main
git push origin main
```

## Step 2: Create Git Tag

```powershell
# Create annotated tag
git tag -a 0.7.3 -m "release: 0.7.3"

# Push tag to remote
git push origin 0.7.3
```

## Step 3: Create GitHub Release

### Option A: Using GitHub Web Interface

1. Go to https://github.com/StellarSara/koharu/releases/new
2. Fill in:
   - **Tag**: Select `0.7.3`
   - **Release title**: `0.7.3`
   - **Description**: Copy content from `RELEASES` file
3. Click "Publish release"

### Option B: Using GitHub CLI

```powershell
# Install GitHub CLI if needed
# winget install GitHub.cli

# Create release with notes from RELEASES file
gh release create 0.7.3 --title "0.7.3" --notes-file RELEASES
```

## Step 3: Build and Upload Binaries

### Build for Windows

```powershell
# Build with CUDA
bun tauri build --features cuda

# Binaries will be in:
# - target/release/Koharu-win-Setup.exe
# - target/release/Koharu-win-Portable.zip
```

### Upload to Release

#### Using GitHub Web Interface:
1. Go to the release page at https://github.com/StellarSara/koharu/releases
2. Click "Edit"
3. Drag and drop the binaries to "Attach binaries"

#### Using GitHub CLI:
```powershell
gh release upload 0.7.3 `
  "target/release/bundle/nsis/Koharu_0.7.3_x64-setup.exe#Koharu-win-Setup.exe" `
  "target/release/bundle/nsis/Koharu_0.7.3_x64.zip#Koharu-win-Portable.zip"
```

## Release Checklist

- [x] Version updated in `Cargo.toml` (0.7.3)
- [ ] All changes committed and pushed
- [ ] Git tag created and pushed
- [ ] GitHub release created
- [ ] Binaries built with CUDA support
- [ ] Binaries uploaded to release
- [ ] Release notes include:
  - ✅ Feature description
  - ✅ Usage instructions
  - ✅ Model setup requirements
  - ✅ Documentation links

## Post-Release Tasks

1. Test the release binaries on a clean Windows machine
2. Update documentation if needed
3. Announce the release on Discord/social media
4. Close any related issues/PRs

## Notes

- Make sure PaddleOCR models are exported and uploaded to Hugging Face before users try the release
- Update `paddle-ocr/src/lib.rs` with correct Hugging Face repo name
- Consider creating a separate release for model files if needed
