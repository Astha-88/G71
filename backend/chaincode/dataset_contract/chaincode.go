package main

import (
	"encoding/json"
	"fmt"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// DatasetContract implements dataset governance chaincode.
type DatasetContract struct {
	contractapi.Contract
}

type Dataset struct {
	DatasetID   string `json:"dataset_id"`
	HospitalID  string `json:"hospital_id"`
	Disease     string `json:"disease"`
	DatasetHash string `json:"dataset_hash"`
	StoragePath string `json:"storage_path"`
}

type AccessRequest struct {
	RequestID   string   `json:"request_id"`
	DatasetID   string   `json:"dataset_id"`
	RequesterID string   `json:"requester_id"`
	Status      string   `json:"status"` // pending, approved, rejected
	Approvals   []string `json:"approvals"`
}

func (c *DatasetContract) RegisterDataset(ctx contractapi.TransactionContextInterface, datasetJSON string) error {
	var ds Dataset
	if err := json.Unmarshal([]byte(datasetJSON), &ds); err != nil {
		return fmt.Errorf("invalid dataset payload: %w", err)
	}
	key := "dataset_" + ds.DatasetID
	existing, err := ctx.GetStub().GetState(key)
	if err != nil {
		return err
	}
	if existing != nil {
		return fmt.Errorf("dataset already exists")
	}
	bytes, err := json.Marshal(ds)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, bytes)
}

func (c *DatasetContract) RequestDatasetAccess(ctx contractapi.TransactionContextInterface, requestJSON string) error {
	var req AccessRequest
	if err := json.Unmarshal([]byte(requestJSON), &req); err != nil {
		return fmt.Errorf("invalid request payload: %w", err)
	}
	req.Status = "pending"
	key := "request_" + req.RequestID
	bytes, err := json.Marshal(req)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, bytes)
}

func (c *DatasetContract) ApproveAccess(ctx contractapi.TransactionContextInterface, requestID string, approverID string) error {
	key := "request_" + requestID
	bytes, err := ctx.GetStub().GetState(key)
	if err != nil {
		return err
	}
	if bytes == nil {
		return fmt.Errorf("request not found")
	}
	var req AccessRequest
	if err := json.Unmarshal(bytes, &req); err != nil {
		return err
	}
	req.Approvals = append(req.Approvals, approverID)
	updated, err := json.Marshal(req)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, updated)
}

func (c *DatasetContract) RejectAccess(ctx contractapi.TransactionContextInterface, requestID string) error {
	key := "request_" + requestID
	bytes, err := ctx.GetStub().GetState(key)
	if err != nil {
		return err
	}
	if bytes == nil {
		return fmt.Errorf("request not found")
	}
	var req AccessRequest
	if err := json.Unmarshal(bytes, &req); err != nil {
		return err
	}
	req.Status = "rejected"
	updated, err := json.Marshal(req)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, updated)
}

func (c *DatasetContract) GetDataset(ctx contractapi.TransactionContextInterface, datasetID string) (*Dataset, error) {
	key := "dataset_" + datasetID
	bytes, err := ctx.GetStub().GetState(key)
	if err != nil {
		return nil, err
	}
	if bytes == nil {
		return nil, fmt.Errorf("dataset not found")
	}
	var ds Dataset
	if err := json.Unmarshal(bytes, &ds); err != nil {
		return nil, err
	}
	return &ds, nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(new(DatasetContract))
	if err != nil {
		panic(err)
	}
	if err := chaincode.Start(); err != nil {
		panic(err)
	}
}

