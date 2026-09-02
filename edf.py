import numpy as np


def edfread(fname, assignToVariables=False, targetSignals=None):
    # Open the EDF file
    try:
        fid = open(fname, "rb")  # chars till 5600
    except IOError as e:
        raise IOError(f"Error opening file: {e}")

    # HEADER
    hdr = {}
    hdr["ver"] = int(fid.read(8).decode("ascii").strip())
    hdr["patientID"] = fid.read(80).decode("ascii").strip()
    hdr["recordID"] = fid.read(80).decode("ascii").strip()
    hdr["startdate"] = fid.read(8).decode("ascii").strip()
    hdr["starttime"] = fid.read(8).decode("ascii").strip()
    hdr["bytes"] = int(fid.read(8).decode("ascii").strip())
    reserved = fid.read(44)
    hdr["records"] = int(fid.read(8).decode("ascii").strip())
    hdr["duration"] = float(fid.read(8).decode("ascii").strip())
    hdr["ns"] = int(fid.read(4).decode("ascii").strip())

    hdr["label"] = [fid.read(16).decode("ascii").strip() for _ in range(hdr["ns"])]

    # Handle targetSignals (subset of signals to read)
    if targetSignals is None:
        targetSignals = list(range(hdr["ns"]))  # Read all signals by default
    elif isinstance(targetSignals, (str, list)):
        targetSignals = [hdr["label"].index(signal) for signal in targetSignals]

    hdr["transducer"] = [fid.read(80).decode("ascii").strip() for _ in range(hdr["ns"])]
    hdr["units"] = [fid.read(8).decode("ascii").strip() for _ in range(hdr["ns"])]
    hdr["physicalMin"] = np.array(
        [float(fid.read(8).decode("ascii").strip()) for _ in range(hdr["ns"])]
    )
    hdr["physicalMax"] = np.array(
        [float(fid.read(8).decode("ascii").strip()) for _ in range(hdr["ns"])]
    )
    hdr["digitalMin"] = np.array(
        [float(fid.read(8).decode("ascii").strip()) for _ in range(hdr["ns"])]
    )
    hdr["digitalMax"] = np.array(
        [float(fid.read(8).decode("ascii").strip()) for _ in range(hdr["ns"])]
    )
    hdr["prefilter"] = [fid.read(80).decode("ascii").strip() for _ in range(hdr["ns"])]
    hdr["samples"] = np.array(
        [int(fid.read(8).decode("ascii").strip()) for _ in range(hdr["ns"])]
    )

    # Read reserved bytes for each signal
    for _ in range(hdr["ns"]):
        fid.read(32).decode("ascii").strip()

    hdr["label"] = [hdr["label"][i] for i in targetSignals]

    print("Step 1 of 2: Reading requested records. (This may take a few minutes)...")

    if assignToVariables or len(targetSignals) > 0:
        # Scaling factors for the signals
        scalefac = (hdr["physicalMax"] - hdr["physicalMin"]) / (
            hdr["digitalMax"] - hdr["digitalMin"]
        )
        dc = hdr["physicalMax"] - scalefac * hdr["digitalMax"]

        # Read records
        record = np.zeros((len(targetSignals), hdr["samples"][0] * hdr["records"]))

        for recnum in range(hdr["records"]):
            for ii in range(hdr["ns"]):
                if ii in targetSignals:
                    num_samples = hdr["samples"][ii]
                    data = np.fromfile(fid, dtype=np.int16, count=num_samples)
                    data = data * scalefac[ii] + dc[ii]
                    record[ii, recnum * num_samples : (recnum + 1) * num_samples] = data
                else:
                    fid.seek(hdr["samples"][ii] * 2, 1)  # Skip samples we don't need

    print("Step 2 of 2: Parsing data...")

    if assignToVariables:
        locals().update(
            {hdr["label"][i]: record[i, :] for i in range(len(hdr["label"]))}
        )
        return hdr, None

    fid.close()
    return hdr, record
